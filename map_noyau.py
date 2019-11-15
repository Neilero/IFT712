# -*- coding: utf-8 -*-

#####
# Aurélien Vauthier (19 126 456)
# Sahar Tahir (19 145 088)
# Ikram Mekkid (19 143 008)
###

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform, euclidean
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm


class MAPnoyau:
    def __init__(self, lamb=0.2, sigma_square=1.06, b=1.0, c=0.1, d=1.0, M=2, noyau='rbf'):
        """
        Classe effectuant de la segmentation de données 2D 2 classes à l'aide de la méthode à noyau.

        lamb: coefficiant de régularisation L2
        sigma_square: paramètre du noyau rbf
        b, d: paramètres du noyau sigmoidal
        M,c: paramètres du noyau polynomial
        noyau: rbf, lineaire, polynomial ou sigmoidal
        """
        self.lamb = lamb
        self.a = None
        self.sigma_square = sigma_square
        self.M = M
        self.c = c
        self.b = b
        self.d = d
        self.noyau = noyau
        self.x_train = None
        self.kernel = None      # evite la redefinition du noyau
        self.rangeDic = {       # valeurs des hyper-parametres teste en cross-validation
            "sigma": np.linspace(1, 5, num=25),     # la borne supperieur etant toujours prise, nous avons deplace l'interval
            "lamb": np.geomspace(0.000000001, 2, num=10),
            "c": np.linspace(0, 5, num=10),
            "b": np.geomspace(0.001, 0.1, num=10),
            "d": np.geomspace(0.5, 1.5, num=10),    # la borne supperieur etant toujours prise, nous avons deplace l'interval
            "M": range(2, 7)
        }

    def entrainement(self, x_train, t_train):
        """
        Entraîne une méthode d'apprentissage à noyau de type Maximum a
        posteriori (MAP) avec un terme d'attache aux données de type
        "moindre carrés" et un terme de lissage quadratique (voir
        Eq.(1.67) et Eq.(6.2) du livre de Bishop).  La variable x_train
        contient les entrées (un tableau 2D Numpy, où la n-ième rangée
        correspond à l'entrée x_n) et des cibles t_train (un tableau 1D Numpy
        où le n-ième élément correspond à la cible t_n).

        L'entraînement doit utiliser un noyau de type RBF, lineaire, sigmoidal,
        ou polynomial (spécifié par ''self.noyau'') et dont les parametres
        sont contenus dans les variables self.sigma_square, self.c, self.b, self.d
        et self.M et un poids de régularisation spécifié par ``self.lamb``.

        Cette méthode doit assigner le champs ``self.a`` tel que spécifié à
        l'equation 6.8 du livre de Bishop et garder en mémoire les données
        d'apprentissage dans ``self.x_train``
        """

        self.x_train = x_train

        if self.kernel is None:
            if self.noyau == "lineaire":
                self.kernel = lambda x1,x2 : np.transpose(x1).dot(x2)

            elif self.noyau == "rbf":
                self.kernel = lambda x1,x2 : np.exp( - (euclidean(x1, x2)**2) / (2*self.sigma_square) )

            elif self.noyau == "polynomial":
                self.kernel = lambda x1,x2 : (np.transpose(x1).dot(x2) + self.c)** self.M

            elif self.noyau == "sigmoidal":
                self.kernel = lambda x1,x2 : np.tanh( self.b * np.transpose(x1).dot(x2) + self.d )

            else:
                raise ValueError("Noyau inconnu")

        K = squareform(pdist(x_train, self.kernel))
        self.a = np.dot(np.linalg.inv(K + (self.lamb*np.identity(len(x_train))) ), t_train)
        
    def prediction(self, x):
        """
        Retourne la prédiction pour une entrée representée par un tableau
        1D Numpy ``x``.

        Cette méthode suppose que la méthode ``entrainement()`` a préalablement
        été appelée. Elle doit utiliser le champs ``self.a`` afin de calculer
        la prédiction y(x) (équation 6.9).

        NOTE : Puisque nous utilisons cette classe pour faire de la
        classification binaire, la prediction est +1 lorsque y(x)>0.5 et 0
        sinon
        """

        prediction = np.dot(cdist(self.x_train, np.atleast_2d(x), self.kernel).transpose(), self.a)
        return np.array(prediction > 0.5, dtype=float)

    def erreur(self, t, prediction):
        """
        Retourne la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``.
        """

        return np.power((t-prediction), 2)

    def validation_croisee(self, x_tab, t_tab):
        """
        Cette fonction trouve les meilleurs hyperparametres ``self.sigma_square``,
        ``self.c`` et ``self.M`` (tout dépendant du noyau selectionné) et
        ``self.lamb`` avec une validation croisée de type "k-fold" où k=1 avec les
        données contenues dans x_tab et t_tab.  Une fois les meilleurs hyperparamètres
        trouvés, le modèle est entraîné une dernière fois.

        SUGGESTION: Les valeurs de ``self.sigma_square`` et ``self.lamb`` à explorer vont
        de 0.000000001 à 2, les valeurs de ``self.c`` de 0 à 5, les valeurs
        de ''self.b'' et ''self.d'' de 0.00001 à 0.01 et ``self.M`` de 2 à 6
        """
        best_error = float("inf")

        if self.noyau == "lineaire":
            self.lamb, best_error = self.recherche_meilleur_lamb(x_tab, t_tab)
            print("best hyper-parameter found: lamb = {} (error = {})".format(self.lamb, best_error))

        elif self.noyau == "rbf":
            best_sigma = self.sigma_square
            best_lamb = self.lamb

            for sigma in tqdm(self.rangeDic["sigma"]):
                self.sigma_square = sigma

                lamb, lamb_error = self.recherche_meilleur_lamb(x_tab, t_tab)
                if lamb_error < best_error:
                    best_error = lamb_error
                    best_sigma = sigma
                    best_lamb = lamb

            print("best hyper-parameters found : sigma = {} and lamb = {} (error = {})".format(best_sigma, best_lamb, best_error))
            self.sigma_square = best_sigma
            self.lamb = best_lamb

        elif self.noyau == "polynomial":
            best_c = self.c
            best_M = self.M
            best_lamb = self.lamb

            for c in tqdm(self.rangeDic["c"]):
                self.c = c

                for M in self.rangeDic["M"]:
                    self.M = M

                    lamb, lamb_error = self.recherche_meilleur_lamb(x_tab, t_tab)
                    if lamb_error < best_error:
                        best_error = lamb_error
                        best_c = c
                        best_M = M
                        best_lamb = lamb

            print("best hyper-parameters found : c = {}, M = {} and lamb = {} (error = {})".format(best_c, best_M, best_lamb, best_error))
            self.c = best_c
            self.M = best_M
            self.lamb = best_lamb

        elif self.noyau == "sigmoidal":
            best_b = self.b
            best_d = self.d
            best_lamb = self.lamb

            for b in tqdm(self.rangeDic["b"]):
                self.b = b

                for d in self.rangeDic["d"]:
                    self.d = d

                    lamb, lamb_error = self.recherche_meilleur_lamb(x_tab, t_tab)
                    if lamb_error < best_error:
                        best_error = lamb_error
                        best_b = b
                        best_d = d
                        best_lamb = lamb

            print("best hyper-parameters found : b = {}, d = {} and lamb = {} (error = {})".format(best_b, best_d, best_lamb, best_error))
            self.b = best_b
            self.d = best_d
            self.lamb = best_lamb

        else:
            raise ValueError("Noyau inconnu")

        # entrainement apres cross-validation
        self.entrainement(x_tab, t_tab)

    def recherche_meilleur_lamb(self, x_tab, t_tab):
        """
        Cette fonction trouve le meilleur hyperparametre ``self.lamb`` et le retourne avec son erreur
        """
        k = 1
        validate_size = 0.3
        best_lamb = self.lamb
        best_lamb_error = float("inf")

        for lamb in self.rangeDic["lamb"]:
            self.lamb = lamb
            error = 0
            for train_index, validate_index in ShuffleSplit(n_splits=k, test_size=validate_size).split(x_tab):
                self.entrainement(x_tab[train_index], t_tab[train_index])
                error += np.sum(self.erreur( t_tab[validate_index], self.prediction(x_tab[validate_index]) ))

            if error < best_lamb_error:
                best_lamb_error = error
                best_lamb = lamb

        return best_lamb, best_lamb_error

    def affichage(self, x_tab, t_tab):

        # Affichage
        ix = np.arange(x_tab[:, 0].min(), x_tab[:, 0].max(), 0.1)
        iy = np.arange(x_tab[:, 1].min(), x_tab[:, 1].max(), 0.1)
        iX, iY = np.meshgrid(ix, iy)
        x_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
        contour_out = np.array([self.prediction(x) for x in x_vis])
        contour_out = contour_out.reshape(iX.shape)

        plt.contourf(iX, iY, contour_out > 0.5)
        plt.scatter(x_tab[:, 0], x_tab[:, 1], s=(t_tab + 0.5) * 100, c=t_tab, edgecolors='y')
        plt.show()
