from tqdm import tqdm
import torch
from src.parse_config import ConfigParser
from pathlib import Path
import os
import torchaudio
from torch.nn.utils.rnn import pad_sequence


class Trainer:

    def __init__(self, config: ConfigParser):
        self.config = config
        self.device = config.get_device()
        self.model = config.get_model()
        self.MSD, self.MPD = config.get_discriminators()
        self.do_val = config['trainer']['do_val']
        if self.do_val:
            self.train_dataloader, self.val_dataloader = config.get_dataloaders()
        else:
            self.train_dataloader, _ = config.get_dataloaders()
        self.writer = config.get_writer()
        self.logger = config.get_logger('trainer')
        self.gen_opt, self.disc_opt = config.get_optimizers(self.model, self.MSD, self.MPD)
        self.start_epoch = 1
        self.best_loss = 1000
        self.do_save = config['trainer']['do_save']
        self.sr = config['melspectrogram']['sample_rate']

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        self.gen_scheduler, self.disc_scheduler = config.get_schedulers(self.gen_opt, self.disc_opt)
        self.melspectrogram = config.get_melspectrogram()
        self.gen_crit, self.disc_crit = config.get_criterions()
        self.overfit = config['trainer']['overfit']

        self.n_epoch = config['trainer']['n_epoch']
        if self.overfit:
            self.len_epoch = config['trainer']['len_epoch']
            if self.do_val:
                self.val_len_epoch = config['trainer']['val_len_epoch']
        else:
            self.len_epoch = min(config['trainer']['len_epoch'],
                                 len(self.train_dataloader.dataset) // config['data']['train']['batch_size'])
            if self.do_val:
                self.val_len_epoch = min(config['trainer']['val_len_epoch'],
                                         len(self.val_dataloader.dataset) // config['data']['val']['batch_size'])

        self.do_test = False
        if 'test' in self.config['data']:
            self.do_test = True
            test_audio = []
            testdir = Path(self.config['data']['test']['dir'])
            self.test_names = os.listdir(testdir)
            for file in self.test_names:
                audio, sr = torchaudio.load(testdir / file)
                test_audio.append(audio.squeeze(0))

            test_wavs = pad_sequence(test_audio).transpose(0, 1).to(self.device)
            with torch.no_grad():
                self.test_specs = self.melspectrogram(test_wavs)

        self.log_audio_interval = config['trainer']['log_audio_interval']
        self.checkpoint_dir = config.save_dir
        self.save_period = config['trainer']['save_period']

        self.logger.info(self.model)
        self.logger.info(self.MPD)
        self.logger.info(self.MSD)
        self.logger.info('Generator number of parameters {}'.format(sum(p.numel() for p in self.model.parameters())))
        self.logger.info('MPD number of parameters {}'.format(sum(p.numel() for p in self.MPD.parameters())))
        self.logger.info('MSD number of parameters {}'.format(sum(p.numel() for p in self.MSD.parameters())))

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _process_batch(self, batch, epoch_num, batch_idx):
        self.writer.set_step((epoch_num - 1) * self.len_epoch + batch_idx, 'train')

        waveform = batch['waveform'].to(self.device)
        spec = self.melspectrogram(waveform)

        # add channel dim
        waveform = waveform.unsqueeze(1)
        pred_wav = self.model(spec)
        pred_spec = self.melspectrogram(pred_wav.squeeze(1))

        # train discriminators
        self.disc_opt.zero_grad()

        # MPD
        mpd_pred, _ = self.MPD(pred_wav.detach())
        mpd_true, _ = self.MPD(waveform)

        # MSD
        msd_pred, _ = self.MSD(pred_wav.detach())
        msd_true, _ = self.MSD(waveform)

        disc_loss = self.disc_crit(mpd_pred, mpd_true, msd_pred, msd_true)

        disc_loss.backward()
        self.disc_opt.step()

        # train generator
        self.gen_opt.zero_grad()

        # MPD
        mpd_pred, mpd_pred_feats = self.MPD(pred_wav)
        _, mpd_true_feats = self.MPD(waveform)

        # MSD
        msd_pred, msd_pred_feats = self.MSD(pred_wav)
        _, msd_true_feats = self.MSD(waveform)

        gen_loss = self.gen_crit(spec, pred_spec, mpd_pred_feats, mpd_true_feats, msd_pred_feats, msd_true_feats,
                                 mpd_pred, msd_pred)

        gen_loss.backward()
        self.gen_opt.step()

        self.disc_scheduler.step()
        self.gen_scheduler.step()

        self.writer.add_scalar("Discriminators Loss", disc_loss)
        self.writer.add_scalar("Generator Loss", gen_loss)
        if batch_idx % self.log_audio_interval == 0:
            self.writer.add_image('True spec', spec[0].detach().cpu().numpy(), dataformats='HW')
            self.writer.add_image('Pred spec', pred_spec[0].detach().cpu().numpy(), dataformats='HW')
            self.writer.add_audio('True audio', waveform[0], sample_rate=self.sr)
            self.writer.add_audio('Pred audio', pred_wav[0], sample_rate=self.sr)

    def _train_epoch(self, num):
        self.model.train()
        self.MPD.train()
        self.MSD.train()
        for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc=f'Epoch {num}', total=self.len_epoch)):
            if batch_idx >= self.len_epoch:
                break
            self._process_batch(batch, num, batch_idx)

    def _val_epoch(self, num):
        self.model.eval()
        self.MPD.eval()
        self.MSD.eval()
        self.writer.set_step(num * self.len_epoch, 'val')

        if self.do_test:
            with torch.no_grad():
                self.logger.info("Synthesizing test audio...")
                pred_wav = self.model(self.test_specs).cpu()
                self.logger.info("Test audio synthesized. Saving...")
                sr = self.config['melspectrogram']['sample_rate']
                save_dir = self.config.save_dir / f'epoch{num}'
                save_dir.mkdir(parents=True, exist_ok=True)
                for i in range(len(pred_wav)):
                    path = save_dir / f'Synthesized_{self.test_names[i]}.wav'
                    torchaudio.save(path, pred_wav[i], sr)
                    self.writer.add_audio(f'Test audio {self.test_names[i]}', path, sr)

        if not self.do_val:
            return 0

        disc_loss_sum = 0
        gen_loss_sum = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_dataloader, desc=f'Validation', total=self.val_len_epoch)):
                if batch_idx >= self.val_len_epoch:
                    break
                waveform = batch['waveform'].to(self.device)
                spec = self.melspectrogram(waveform)

                waveform = waveform.unsqueeze(1)
                pred_wav = self.model(spec)
                pred_spec = self.melspectrogram(pred_wav.squeeze(1))

                # MPD
                mpd_pred, mpd_pred_feats = self.MPD(pred_wav)
                mpd_true, mpd_true_feats = self.MPD(waveform)

                # MSD
                msd_pred, msd_pred_feats = self.MSD(pred_wav)
                msd_true, msd_true_feats = self.MSD(waveform)

                disc_loss = self.disc_crit(mpd_pred, mpd_true, msd_pred, msd_true)
                gen_loss = self.gen_crit(spec, pred_spec, mpd_pred_feats, mpd_true_feats, msd_pred_feats,
                                         msd_true_feats, mpd_pred, msd_pred)

                disc_loss_sum += disc_loss
                gen_loss_sum += gen_loss

        self.writer.add_image('True spec', spec[0].detach().cpu().numpy(), dataformats='HW')
        self.writer.add_image('Pred spec', pred_spec[0].detach().cpu().numpy(), dataformats='HW')
        self.writer.add_audio('True audio', waveform[0], sample_rate=self.sr)
        self.writer.add_audio('Pred audio', pred_wav[0], sample_rate=self.sr)

        disc_loss_sum /= batch_idx + 1
        gen_loss_sum /= batch_idx + 1
        self.writer.add_scalar("Discriminators Loss", disc_loss_sum)
        self.writer.add_scalar("Generator Loss", gen_loss_sum)

        return gen_loss_sum

    def _train_process(self):
        for epoch_num in range(self.start_epoch, self.n_epoch + 1):
            self._last_epoch = epoch_num
            self._train_epoch(epoch_num)
            best = False
            loss_avg = self._val_epoch(epoch_num)
            if self.do_val:
                self.logger.info('Val gen loss: {}'.format(loss_avg))
                if loss_avg < self.best_loss:
                    self.best_loss = loss_avg
                    best = True
            if self.do_save and (epoch_num % self.save_period == 0 or best):
                self._save_checkpoint(epoch_num, save_best=best, only_best=True)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "MSD": self.MSD.state_dict(),
            "MPD": self.MPD.state_dict(),
            "gen_optimizer": self.gen_opt.state_dict(),
            "disc_optimizer": self.disc_opt.state_dict(),
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        if not (only_best and save_best):
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1

        # load architecture params from checkpoint.
        self.model.load_state_dict(checkpoint["state_dict"])

        self.gen_opt.load_state_dict(checkpoint["gen_optimizer"])
        self.disc_opt.load_state_dict(checkpoint["disc_optimizer"])
        self.MSD.load_state_dict(checkpoint['MSD'])
        self.MPD.load_state_dict(checkpoint['MPD'])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
