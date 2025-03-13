#Copyright (c) 2017 Ruey-Cheng Chen

""" This is a modified version of the AdaRank algorithm. The original code was copied from the repository git@github.com:rueycheng/AdaRank.git.

 CHANGES MADE: 
1. Added EPS during alpha calculation to avoid division by zero
2. Added prints 
3. Add get_coef function to return the coefficients of the model
4. Explicit regularization using lambda param to enforce feature diversity during model fit -> Prohibits AdaRank to only select most important feature
5. Radonmize Weak Ranker Iteration order during fit to avoid feature being overmephazised due to order if similar predition power 
6. Add a max alpha to avoid overfitting

AdaRank algorithm
"""
from __future__ import print_function, division

import math
import numpy as np
import sklearn
import sys
import random

from sklearn.utils import check_X_y

from metrics import NDCGScorer


class AdaRankv2(sklearn.base.BaseEstimator):
    """AdaRank algorithm"""

    def __init__(self, max_iter=500, tol=0.0001, estop=1, verbose=False, scorer=None):
        self.max_iter = max_iter
        self.tol = tol
        self.estop = estop
        self.verbose = verbose
        self.scorer = scorer

    def fit(self, X, y, qid, X_valid=None, y_valid=None, qid_valid=None):
        """Fit a model to the data"""
        X, y = check_X_y(X, y, 'csr')
        X = X.toarray()

        if X_valid is None:
            X_valid, y_valid, qid_valid = X, y, qid
        else:
            X_valid, y_valid = check_X_y(X_valid, y_valid, 'csr')
            X_valid = X_valid.toarray()

        n_queries = np.unique(qid).shape[0]
        weights = np.ones(n_queries, dtype=np.float64) / n_queries
        weak_rankers = []
        coef = np.zeros(X.shape[1])

        # use nDCG@10 as the default scorer
        if self.scorer is None:
            self.scorer = NDCGScorer(k=10)

        # precompute performance measurements for all weak rankers
        weak_ranker_score = []
        for j in range(X.shape[1]):
            pred = X[:, j].ravel()
            weak_ranker_score.append(self.scorer(y, pred, qid))

        best_perf_train = -np.inf
        best_perf_valid = -np.inf
        used_fids = []
        estop = None

        self.n_iter = 0
        while self.n_iter < self.max_iter:
            self.n_iter += 1

            best_weighted_average = -np.inf
            best_weak_ranker = None
            #CHANGED CODE: Rnadomize the order of weak ranker iteration
            # Create a list of all feature indices
            candidate_fids = list(range(len(weak_ranker_score)))
            # Randomize the order of candidates
            random.shuffle(candidate_fids)

            for fid in candidate_fids:
                score = weak_ranker_score[fid]
                # CHANGED CODE: Apply a penalty if this feature was used before
                #   penalty = 0.5 if fid in used_fids else 1.0
                #   weighted_average = penalty * np.dot(weights, score)
                weighted_average = np.dot(weights, score)
                if weighted_average > best_weighted_average:
                    best_weak_ranker = {'fid': fid, 'score': score}
                    best_weighted_average = weighted_average

            # stop when no candidate is found
            if best_weak_ranker is None:
                break

            #CHANGED CODE: Add small epsilon to avoid division by zero
            EPS = 1e-10
            #CHANGED CODE: Add lambda to enforce regulatrization during on the feature selection
            lambda_reg = 0.1 # hyperparam was manually tuned

            h = best_weak_ranker
            # Compute numerator and denominator with regularization added
            num_val = np.dot(weights, 1 + h['score']) + lambda_reg
            den_val = np.dot(weights, 1 - h['score']) + lambda_reg
            # Ensure neither term is below EPS
            num_val = max(num_val, EPS)
            den_val = max(den_val, EPS)
            alpha_calculated = 0.5 * math.log(num_val / den_val)
            max_alpha = 10
            h['alpha'] = min(alpha_calculated, max_alpha)
            
            weak_rankers.append(h)

            # update the ranker
            coef[h['fid']] += h['alpha']

            if len(used_fids) > 5:
                used_fids.pop(0)
            used_fids.append(h['fid'])

            # score both training and validation data
            score_train = self.scorer(y, np.dot(X, coef), qid)
            perf_train = score_train.mean()

            perf_valid = perf_train
            if X_valid is not X:
                perf_valid = self.scorer(y_valid, np.dot(X_valid, coef), qid_valid).mean()

            if self.verbose:
                print('{n_iter}\t{alpha}\t{fid}\t{score}\ttrain {train:.4f}\tvalid {valid:.4f}'.
                      format(n_iter=self.n_iter, alpha=h['alpha'], fid=h['fid'],
                             score=h['score'][:5], train=perf_train, valid=perf_valid),
                      file=sys.stderr)

            # update the best validation scores
            if perf_valid > best_perf_valid + self.tol:
                estop = 0
                best_perf_valid = perf_valid
                self.coef_ = coef.copy()
            else:
                estop += 1

            # update the best training score
            if perf_train > best_perf_train + self.tol:
                best_perf_train = perf_train
            else:
                # stop if scores on both sets fail to improve
                if estop >= self.estop:
                    break

            # update weights
            new_weights = np.exp(-score_train)
            weights = new_weights / new_weights.sum()

        return self

    def predict(self, X, qid):
        """Make predictions"""
        return np.dot(X.toarray(), self.coef_)
    
    def get_coef(self):
        """Return the coefficients of the model"""
        return self.coef_
