# This project uses the structure of MUSE (https://github.com/facebookresearch/MUSE)

import os
import time
import json
import argparse
from collections import OrderedDict
import numpy as np
import torch
from tqdm import tqdm
from unsup_align.src.utils import bool_flag, initialize_exp
from unsup_align.src.models import build_model
from unsup_align.src.trainer import Trainer
from unsup_align.src.evaluation import Evaluator
from unsup_align.src.refinement import generate_new_dictionary, symmetric_reweighting
from unsup_align.src.evaluation.word_translation import get_word_translation_accuracy

VALIDATION_METRIC_AB = 'mean_cosine-csls_knn_10-S2T-10000'
VALIDATION_METRIC_BA = 'mean_cosine-csls_knn_10-T2S-10000'

# build model / trainer / evaluator
class getalign():
    def __init__(self,exp_folder,send,receive,send_emb_path,receive_emb_path):
        # main
        parser = argparse.ArgumentParser(description='Unsupervised training')
        parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
        parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
        parser.add_argument("--exp_path", type=str, default=exp_folder, help="Where to store experiment logs and models")
        parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
        parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
        parser.add_argument("--cuda", type=bool_flag, default=False, help="Run on GPU")
        parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
        # data
        parser.add_argument("--src_lang", type=str, default=send, help="Source language")
        parser.add_argument("--tgt_lang", type=str, default=receive, help="Target language")
        parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
        parser.add_argument("--emb_dim_autoenc", type=int, default=200,
                            help="Embedding dimension in bottle kneck of autoencoder")
        parser.add_argument("--max_vocab_A", type=int, default=200000,
                            help="Maximum vocabulary size for source (-1 to disable)")
        parser.add_argument("--max_vocab_B", type=int, default=200000,
                            help="Maximum vocabulary size for target (-1 to disable)")
        # mapping
        parser.add_argument("--map_id_init", type=bool_flag, default=True,
                            help="Initialize the mapping as an identity matrix")
        parser.add_argument("--map_beta", type=float, default=0.01, help="Beta for orthogonalization")
        # discriminator
        parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
        parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
        parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
        parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
        parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
        parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
        parser.add_argument("--cycle_lambda", type=float, default=5, help="Cycle loss feedback coefficient")
        parser.add_argument("--reconstruction_lambda", type=float, default=1,
                            help="Reconstruction loss feedback coefficient")
        parser.add_argument("--dis_most_frequent_AB", type=int, default=0,
                            help="Select embeddings of the k most frequent words for discrimination in source to target (0 to disable)")
        parser.add_argument("--dis_most_frequent_BA", type=int, default=0,
                            help="Select embeddings of the k most frequent words for discrimination in target to source (0 to disable)")
        parser.add_argument("--dis_smooth", type=float, default=0.2, help="Discriminator smooth predictions")
        parser.add_argument("--dis_clip_weights", type=float, default=0,
                            help="Clip discriminator weights (0 to disable)")
        # training adversarial
        parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
        parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
        parser.add_argument("--epoch_size", type=int, default=10000, help="Iterations per epoch")
        parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
        parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
        parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
        parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
        parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
        parser.add_argument("--lr_shrink", type=float, default=0.5,
                            help="Shrink the learning rate if the validation metric decreases (1 to disable)")
        # training refinement
        parser.add_argument("--n_procrustes", type=int, default=5,
                            help="Maximum number of procurstes iterations in refinement procedure")
        parser.add_argument("--n_symmetric_reweighting", type=int, default=20,
                            help="Maximum number of symmetric reweighting iterations in refinement procedure")
        parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
        parser.add_argument("--dico_method", type=str, default='csls_knn_10',
                            help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
        parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
        parser.add_argument("--dico_threshold", type=float, default=0,
                            help="Threshold confidence for dictionary generation")
        parser.add_argument("--dico_max_rank", type=int, default=0,
                            help="Maximum dictionary words rank (0 to disable)")
        parser.add_argument("--dico_min_size", type=int, default=0,
                            help="Minimum generated dictionary size (0 to disable)")
        parser.add_argument("--dico_max_size", type=int, default=15000,
                            help="Maximum generated dictionary size (0 to disable)")
        # reload pre-trained embeddings
        parser.add_argument("--src_emb", type=str, default=send_emb_path, help="Reload source embeddings")
        parser.add_argument("--tgt_emb", type=str, default=receive_emb_path, help="Reload target embeddings")
        parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
        # Autoencoder
        parser.add_argument("--autoenc_epochs", type=int, default=25, help="Number of initail autoencoder epochs")
        parser.add_argument("--autoenc_optimizer", type=str, default="adam,lr=0.001", help="Autoencoder optimizer")
        parser.add_argument("--out_tanh", type=int, default=0,
                            help="Whether output layer of autoenconder use tanh activation")
        parser.add_argument("--l_relu", type=int, default=0,
                            help="Whether bottleneck of autoenconder use leaky relu activation")

        # parse parameters
        params = parser.parse_args()
        # check parameters
        assert not params.cuda or torch.cuda.is_available()
        assert 0 <= params.dis_dropout < 1
        assert 0 <= params.dis_input_dropout < 1
        assert 0 <= params.dis_smooth < 0.5
        assert params.dis_lambda > 0 and params.dis_steps > 0
        assert 0 < params.lr_shrink <= 1
        assert os.path.isfile(params.src_emb)
        assert os.path.isfile(params.tgt_emb)
        assert params.dico_eval == 'default' or params.dico_eval == 'vecmap' or os.path.isfile(params.dico_eval)
        assert params.export in ["", "txt", "pth"]
        self.params = params

    def run(self):
        logger = initialize_exp(self.params)
        src_emb, tgt_emb, mapping_G, mapping_F, discriminator_A, discriminator_B, encoder_A, decoder_A, encoder_B, decoder_B = build_model(
            self.params, True)
        trainer = Trainer(src_emb, tgt_emb, mapping_G, mapping_F, discriminator_A, discriminator_B, encoder_A, decoder_A,
                          encoder_B, decoder_B, self.params)
        evaluator = Evaluator(trainer)

        """
        Learning loop for Adversarial Training
        """
        if self.params.adversarial:

            # first train the autoencoder to become mature
            trainer.train_autoencoder_A()
            trainer.train_autoencoder_B()

            logger.info('----> ADVERSARIAL TRAINING <----\n\n')

            # adversarial training loop
            for n_epoch in range(self.params.n_epochs):

                logger.info('Starting adversarial training epoch %i...' % n_epoch)
                tic = time.time()
                n_words_proc_G = 0
                n_words_proc_F = 0
                stats = {'DIS_COSTS_A': [], 'DIS_COSTS_B': []}

                for n_iter in tqdm(range(0, self.params.epoch_size, self.params.batch_size)):
                    # discriminator training
                    for _ in range(self.params.dis_steps):
                        trainer.dis_step_B(stats)
                        trainer.dis_step_A(stats)

                    # mapping training
                    n_words_proc_G += trainer.mapping_step_G(stats)
                    n_words_proc_F += trainer.mapping_step_F(stats)

                    # log stats
                    if n_iter % 500 == 0:
                        stats_str = [('DIS_COSTS_B', 'Discriminator B loss')]
                        stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                                     for k, v in stats_str if len(stats[k]) > 0]
                        stats_log.append('%i samples/s' % int(n_words_proc_G / (time.time() - tic)))

                        n_words_proc_G = 0
                        for k, _ in stats_str:
                            del stats[k][:]

                        stats_str = [('DIS_COSTS_A', 'Discriminator A loss')]
                        stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                                     for k, v in stats_str if len(stats[k]) > 0]
                        stats_log.append('%i samples/s' % int(n_words_proc_F / (time.time() - tic)))
                        # reset
                        tic = time.time()
                        n_words_proc_F = 0
                        for k, _ in stats_str:
                            del stats[k][:]

                # model evaluation
                to_log = OrderedDict({'n_epoch': n_epoch})
                evaluator.model_selection_criterion(to_log)

                # JSON log / save best model / end of epoch
                logger.info("__log__:%s" % json.dumps(to_log))
                trainer.save_best_AB(to_log, VALIDATION_METRIC_AB)
                trainer.save_best_BA(to_log, VALIDATION_METRIC_BA)
                logger.info('End of epoch %i.\n\n' % n_epoch)

                # update the learning rate (stop if too small)
                trainer.update_lr(to_log, VALIDATION_METRIC_AB, VALIDATION_METRIC_BA)
                if trainer.map_optimizer_G.param_groups[0]['lr'] < self.params.min_lr or trainer.map_optimizer_F.param_groups[0][
                    'lr'] < self.params.min_lr:
                    logger.info('Learning rate < 1e-6. BREAK.')
                    break

        """
        Learning loop for Refinement Procedure
        """

        # Refinement for language A->B
        print("\n \n Refinement iteration for ", self.params.src_lang, "to ", self.params.tgt_lang, "\n")

        # Reload best model from adversarial training
        trainer.reload_best_AB()
        # trainer.build_dictionary_AB()
        # trainer.save_dico()

        prev_score_mean = 0

        # apply the Procrustes solution for language A->B
        print('**Procrustes solution iteration for  ', self.params.src_lang, "to ", self.params.tgt_lang, ' **')
        for n_iter in range(self.params.n_procrustes):
            trainer.build_dictionary_AB()
            trainer.procrustes_AB()
            # embeddings evaluation
            to_log = OrderedDict({'n_iter': n_iter})
            evaluator.model_selection_criterion(to_log)
            # JSON log / save best model / end of epoch
            logger.info("__log__:%s" % json.dumps(to_log))
            trainer.save_best_AB(to_log, VALIDATION_METRIC_AB)
            logger.info('End of refinement iteration %i.\n\n' % n_iter)
        trainer.reload_best_AB()
        trainer.params.dico_build = 'S2T&T2S'
        trainer.build_dictionary_AB()
        trainer.save_dico()

            # # build a dictionary from aligned embeddings
            # emb1 = (trainer.mapping_G(trainer.encoder_A(trainer.src_emb.weight.data)).data)[0:self.params.dico_max_size]
            # emb2 = (trainer.encoder_B(trainer.tgt_emb.weight.data).data)[0:self.params.dico_max_size]
            # emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
            # emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)
            # all_pairs, all_scores = generate_new_dictionary(emb1, emb2)
            #
            # trainer.dico_AB = all_pairs.cuda() if self.params.cuda else all_pairs
            #
            # score_mean = all_scores[:, 0].mean().item()

            # # checking threshold
            # if score_mean - prev_score_mean >= 1e-06:
            #     prev_score_mean = score_mean
            # elif n_iter > 20:
            #     break

        logger.info('Finished %i procrustes iteration ... for A to B' % n_iter)
        to_log = OrderedDict({'n_iter': n_iter})











        # # Symmetric Reweighting for language A->B
        # print("** Symmetric Reweighting for ", self.params.src_lang, "to ", self.params.tgt_lang, " **")
        # for i in range(self.params.n_symmetric_reweighting):
        #     # seed dictionary from previous step
        #     src_indices = trainer.dico_AB.cpu().numpy()[:, 0]
        #     trg_indices = trainer.dico_AB.cpu().numpy()[:, 1]
        #     xw, zw = symmetric_reweighting(src_emb, tgt_emb, src_indices, trg_indices)
        #
        #     emb1 = xw[0:self.params.dico_max_rank]
        #     emb2 = zw[0:self.params.dico_max_rank]
        #     emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
        #     emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)
        #     all_pairs, all_scores = generate_new_dictionary(emb1, emb2)
        #     trainer.dico_AB = all_pairs.cuda() if self.params.cuda else all_pairs
        #
        #     score_mean = all_scores[:, 0].mean().item()
        #     if score_mean - prev_score_mean >= 1e-06:
        #         prev_score_mean = score_mean
        #     elif i > 10:
        #         break
        #
        # logger.info('Finished %i Symmetric Re-weighting iteration ... for A to B' % i)

        # get_word_translation_accuracy(
        #     self.params.src_dico.lang, self.params.src_dico.word2id, xw,
        #     self.params.tgt_dico.lang, self.params.tgt_dico.word2id, zw,
        #     method='csls_knn_10',
        #     dico_eval=self.params.dico_eval
        # )
        #
        # # Refinement for language B->A
        # print("\n \n Refinement iteration for ", self.params.tgt_lang, "to ", self.params.src_lang, "\n")
        #
        # # Reload best model from adversarial training
        # trainer.reload_best_BA()
        # trainer.build_dictionary_BA()
        #
        # prev_score_mean = 0
        # # apply the Procrustes solution for language B->A
        # print('** Procrustes solution iteration for  ', self.params.tgt_lang, "to ", self.params.src_lang, ' **')
        # for n_iter in range(self.params.n_procrustes):
        #     trainer.procrustes_BA()
        #
        #     # build a dictionary from aligned embeddings
        #     emb1 = ((trainer.encoder_A(trainer.src_emb.weight.data)).data)[0:self.params.dico_max_rank]
        #     emb2 = (trainer.mapping_F(trainer.encoder_B(trainer.tgt_emb.weight.data)).data)[0:self.params.dico_max_rank]
        #     emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
        #     emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)
        #     all_pairs, all_scores = generate_new_dictionary(emb2, emb1)
        #
        #     trainer.dico_BA = all_pairs.cuda() if self.params.cuda else all_pairs
        #
        #     score_mean = all_scores[:, 0].mean().item()
        #     # checking threshold
        #     if score_mean - prev_score_mean >= 1e-06:
        #         prev_score_mean = score_mean
        #     elif n_iter > 20:
        #         break
        #
        # logger.info('Finished %i procrustes iteration ... for B to A' % n_iter)
        # to_log = OrderedDict({'n_iter': n_iter})
        #
        # # Symmetric Reweighting for language B->A
        # print("** Symmetric Reweighting for ", self.params.tgt_lang, "to ", self.params.src_lang, " **")
        # for i in range(self.params.n_symmetric_reweighting):
        #     # seed dictionary from previous step
        #     src_indices = trainer.dico_BA.cpu().numpy()[:, 0]
        #     trg_indices = trainer.dico_BA.cpu().numpy()[:, 1]
        #     zw, xw = symmetric_reweighting(tgt_emb, src_emb, src_indices, trg_indices)
        #
        #     emb1 = xw[0:self.params.dico_max_rank]
        #     emb2 = zw[0:self.params.dico_max_rank]
        #     emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
        #     emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)
        #     all_pairs, all_scores = generate_new_dictionary(emb2, emb1)
        #     trainer.dico_BA = all_pairs.cuda() if self.params.cuda else all_pairs
        #
        #     score_mean = all_scores[:, 0].mean().item()
        #     if score_mean - prev_score_mean >= 1e-06:
        #         prev_score_mean = score_mean
        #     elif i > 10:
        #         break
        #
        # logger.info('Finished %i Symmetric Re-weighting iteration ... for B to A' % i)
        # get_word_translation_accuracy(
        #     self.params.tgt_dico.lang, self.params.tgt_dico.word2id, zw,
        #     self.params.src_dico.lang, self.params.src_dico.word2id, xw,
        #     method='csls_knn_10',
        #     dico_eval=self.params.dico_eval
        # )



if __name__ == "__main__":
    receive = "twitter"
    send = "foursquare"
    receive_embedding_path = "../../dataset/twitter/useremb.pt"
    send_embedding_path = "../../dataset/foursquare/useremb.pt"
    exp_folder = "../../dataset/align"

    unsuper_folder = os.path.join(exp_folder, receive, 'unsuper_files')
    os.makedirs(unsuper_folder, exist_ok=True)
    print("makeunsup")
    train = getalign(unsuper_folder,send,receive,send_embedding_path,receive_embedding_path)
    train.run()


def make_unsuper_align(receive,send,receive_embedding_path,send_embedding_path,
                       exp_folder):

    unsuper_folder = os.path.join(exp_folder, 'unsuper_files')
    os.makedirs(unsuper_folder, exist_ok=True)
    print("makeunsup")
    print(send_embedding_path,receive_embedding_path)
    train = getalign(unsuper_folder,send,receive,send_embedding_path,receive_embedding_path)
    train.run()