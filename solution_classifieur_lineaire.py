# -*- coding: utf-8 -*-

#####
# Aurélien Vauthier (19 126 456)
####

import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt


class ClassifieurLineaire:
    def __init__(self, lamb, methode):
        """
        Algorithmes de classification lineaire

        L'argument ``lamb`` est une constante pour régulariser la magnitude
        des poids w et w_0

        ``methode`` :   1 pour classification generative
                        2 pour Perceptron
                        3 pour Perceptron sklearn
        """
        self.w = np.array([1., 2.]) # paramètre aléatoire
        self.w_0 = -5.              # paramètre aléatoire
        self.lamb = lamb
        self.methode = methode

    def entrainement(self, x_train, t_train):
        """
        Entraîne deux classifieurs sur l'ensemble d'entraînement formé des
        entrées ``x_train`` (un tableau 2D Numpy) et des étiquettes de classe cibles
        ``t_train`` (un tableau 1D Numpy).

        Lorsque self.method = 1 : implémenter la classification générative de
        la section 4.2.2 du libre de Bishop. Cette méthode doit calculer les
        variables suivantes:

        - ``p`` scalaire spécifié à l'équation 4.73 du livre de Bishop.

        - ``mu_1`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.75 du livre de Bishop.

        - ``mu_2`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.76 du livre de Bishop.

        - ``sigma`` matrice de covariance (tableau Numpy 2D) de taille DxD,
                    telle que spécifiée à l'équation 4.78 du livre de Bishop,
                    mais à laquelle ``self.lamb`` doit être ADDITIONNÉ À LA
                    DIAGONALE (comme à l'équation 3.28).

        - ``self.w`` un vecteur (tableau Numpy 1D) de taille D tel que
                    spécifié à l'équation 4.66 du livre de Bishop.

        - ``self.w_0`` un scalaire, tel que spécifié à l'équation 4.67
                    du livre de Bishop.

        lorsque method = 2 : Implementer l'algorithme de descente de gradient
                        stochastique du perceptron avec 1000 iterations

        lorsque method = 3 : utiliser la librairie sklearn pour effectuer une
                        classification binaire à l'aide du perceptron

        """
        if self.methode == 1:  # Classification generative
            print('Classification generative')
            self.p = np.mean(t_train)

            x_train_tn = np.multiply(x_train.transpose(), t_train).transpose()   # tableau des xn * tn
            x_train_2_mask = np.all(np.equal( x_train_tn, [0,0] ), axis=1)     # tableau booléen (True : classe 1)
            x_train_1 = x_train[~x_train_2_mask]     # données d'entrainement de la classe 1
            x_train_2 = x_train[x_train_2_mask]    # données d'entrainement de la classe 0
            self.mu_1 = np.mean(x_train_1, axis=0)
            self.mu_2 = np.mean(x_train_2, axis=0)

            x_train_1_centered = x_train_1 - self.mu_1
            x_train_2_centered = x_train_2 - self.mu_2
            N1 = int(np.sum(t_train))
            N2 = int(np.sum(1-t_train))
            D = x_train.shape[1]
            sigma_1 = np.sum(np.matmul(x_train_1_centered.reshape((N1, D, 1)), x_train_1_centered.reshape((N1, 1, D))), axis=0) / N1
            sigma_2 = np.sum(np.matmul(x_train_2_centered.reshape((N2, D, 1)), x_train_2_centered.reshape((N2, 1, D))), axis=0) / N2
            self.sigma = self.p * sigma_1 + (1-self.p) * sigma_2

            self.w = np.dot (np.linalg.inv(self.sigma), (self.mu_1 - self.mu_2))

            mu_1_t = self.mu_1.transpose()
            mu_2_t = self.mu_2.transpose()
            sigma_inv = np.linalg.inv(self.sigma)
            self.w_0 = -0.5 * np.dot(np.dot(mu_1_t, sigma_inv), self.mu_1)
            self.w_0 += 0.5 * np.dot(np.dot(mu_2_t, sigma_inv), self.mu_2)
            self.w_0 += np.log( self.p / (1-self.p) ) # ln( p(C1)/p(C2) )

        elif self.methode == 2:  # Perceptron + SGD, learning rate = 0.001, nb_iterations_max = 1000
            print('Perceptron')
            # AJOUTER CODE ICI

        else:  # Perceptron + SGD [sklearn] + learning rate = 0.001 + penalty 'l2' voir http://scikit-learn.org/
            print('Perceptron [sklearn]')
            # AJOUTER CODE ICI

        print('w = ', self.w, 'w_0 = ', self.w_0, '\n')

    def prediction(self, x):
        """
        Retourne la prédiction du classifieur lineaire.  Retourne 1 si x est
        devant la frontière de décision et 0 sinon.

        ``x`` est un tableau 1D Numpy

        Cette méthode suppose que la méthode ``entrainement()``
        a préalablement été appelée. Elle doit utiliser les champs ``self.w``
        et ``self.w_0`` afin de faire cette classification.
        """

        return int(self.w_0 + np.dot(self.w.transpose(), x) > 0)

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de classification, i.e.
        1. si la cible ``t`` et la prédiction ``prediction``
        sont différentes, 0. sinon.
        """

        return int(t != prediction)

    def afficher_donnees_et_modele(self, x_train, t_train, x_test, t_test):
        """
        afficher les donnees et le modele

        x_train, t_train : donnees d'entrainement
        x_test, t_test : donnees de test
        """
        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1], s=t_train * 100 + 20, c=t_train)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Training data')

        plt.figure(1)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=t_test * 100 + 20, c=t_test)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Testing data')

        plt.show()

    def parametres(self):
        """
        Retourne les paramètres du modèle
        """
        return self.w_0, self.w
