from tqdm import tqdm
import torch
import torchaudio
from src.parse_config import ConfigParser
import matplotlib.pyplot as plt


class Trainer:

    def __init__(self, config: ConfigParser):
        self.config = config
        self.device = config.get_device()
        self.model = config.get_model()
        self.train_dataloader, self.val_dataloader = config.get_dataloaders()
        self.writer = config.get_writer()
        self.logger = config.get_logger('trainer')
        self.optimizer = config.get_optimizer(self.model)
        self.start_epoch = 1
        self.best_loss = 1000
        self.do_save = config['trainer']['do_save']

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        self.scheduler = config.get_scheduler(self.optimizer)
        self.melspectrogram = config.get_melspectrogram()
        self.criterion = config.get_criterion()
        self.overfit = config['trainer']['overfit']

        self.n_epoch = config['trainer']['n_epoch']
        if self.overfit:
            self.len_epoch = config['trainer']['len_epoch']
            self.val_len_epoch = config['trainer']['val_len_epoch']
        else:
            self.len_epoch = min(config['trainer']['len_epoch'],
                                 len(self.train_dataloader.dataset) // config['data']['train']['batch_size'])
            self.val_len_epoch = min(config['trainer']['val_len_epoch'],
                                     len(self.val_dataloader.dataset) // config['data']['val']['batch_size'])
        self.log_audio_interval = config['trainer']['log_audio_interval']
        self.checkpoint_dir = config.save_dir
        self.save_period = config['trainer']['save_period']

        self.logger.info(self.model)
        self.logger.info('Number of parameters {}'.format(sum(p.numel() for p in self.model.parameters())))

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _process_batch(self, batch, epoch_num, batch_idx, is_train=True):
        if is_train:
            self.writer.set_step((epoch_num - 1) * self.len_epoch + batch_idx, 'train')

        waveform = batch['waveform'].to(self.device)
        # waveform_length = batch['waveform_length']
        spec = self.melspectrogram(waveform)

        self.optimizer.zero_grad()
        pred_wav = self.model(spec)
        pred_spec = self.melspectrogram(pred_wav)
        loss = self.criterion(pred_spec, spec)

        if is_train:
            self.writer.add_scalar("Loss", loss)
            if self.overfit and batch_idx % self.log_audio_interval == 0:
                # if self.config['trainer']['visualize'] == 'wandb':
                self.writer.add_image('True spec', plt.imshow(spec[0].detach().cpu().numpy()))
                self.writer.add_image('Pred spec', plt.imshow(pred_spec[0].detach().cpu().numpy()))
                self.writer.add_audio('True audio', waveform[0],
                                      sample_rate=self.config['melspectrogram']['sample_rate'])
                self.writer.add_audio('Pred audio', pred_wav[0],
                                      sample_rate=self.config['melspectrogram']['sample_rate'])
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return loss

    def _train_epoch(self, num):
        self.model.train()
        for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc=f'Epoch {num}', total=self.len_epoch)):
            if batch_idx >= self.len_epoch:
                break
            self._process_batch(batch, num, batch_idx)

    def _val_epoch(self, num):
        self.model.eval()
        self.writer.set_step(num * self.len_epoch, 'val')
        loss_sum = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_dataloader, desc=f'Validation', total=self.val_len_epoch)):
                if batch_idx >= self.val_len_epoch:
                    break
                loss = self._process_batch(batch, num, batch_idx, is_train=False)
                loss_sum += loss

        loss_sum /= self.val_len_epoch
        self.writer.add_scalar("Loss", loss_sum)

        return loss_sum

    def _train_process(self):
        for epoch_num in range(self.start_epoch, self.n_epoch + 1):
            self._last_epoch = epoch_num
            self._train_epoch(epoch_num)
            best = False
            loss_avg = self._val_epoch(epoch_num)
            self.logger.info('Val loss: {}'.format(loss_avg))
            if loss_avg < self.best_loss:
                self.best_loss = loss_avg
                best = True
            if self.do_save and (epoch_num % self.save_period == 0 or best):
                self._save_checkpoint(epoch_num, save_best=best, only_best=True)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
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

        # load optimizer state from checkpoint only when optimizer type is not changed.
        # if (
        #         checkpoint["config"]["optimizer"] != self.config["optimizer"] or
        #         checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        # ):
        #     self.logger.warning(
        #         "Warning: Optimizer or lr_scheduler given in config file is different "
        #         "from that of checkpoint. Optimizer parameters not being resumed."
        #     )
        # else:
        for param, value in self.config['optimizer']['args'].items():
            if param in checkpoint['optimizer']['param_groups'][0]:
                checkpoint['optimizer']['param_groups'][0][param] = value
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.logger.info(self.optimizer)
        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
