# -*- coding: utf-8 -*-

#####
# Aurélien Vauthier (19 126 456)
# TODO add yours
###

import numpy as np
import random
from sklearn import linear_model


class Regression:
    def __init__(self, lamb, m=1):
        self.lamb = lamb
        self.w = None
        self.M = m

    def fonction_base_polynomiale(self, x):
        """
        Fonction de base qui projette la donnee x vers un espace polynomial tel que mentionne au chapitre 3.
        Si x est un scalaire, alors phi_x sera un vecteur à self.M dimensions : (x^1,x^2,...,x^self.M)
        Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de taille NxM

        NOTE : En mettant phi_x = x, on a une fonction de base lineaire qui fonctionne pour une regression lineaire
        """

        n = 1 # on suppose x est un scalaire et on définit le nombre de lignes de phi_x à 1...
        if not np.isscalar(x):
            #... si ce n'est pas le cas on regarde le nombre de colonnes de x pour définir le nombre de ligne de phi_x
            n = len(x)

        m = self.M+1    # +1 pour w0
        x_repeated = np.repeat(x, m).reshape((n, m))
        phi_x = np.power(x_repeated, np.arange(m))
        return phi_x

    def recherche_hyperparametre(self, X, t):
        """
        Validation croisee de type "k-fold" pour k=10 utilisee pour trouver la meilleure valeur pour
        l'hyper-parametre self.M.

        Le resultat est mis dans la variable self.M

        X: vecteur de donnees
        t: vecteur de cibles
        """


        self.M = 1

    def calcule_parametres_optimal(self, X, t):
        """
        Calcul le vecteur de parametres optimal selon les donnees ``X``
        et le vecteur de cible ``t`` via une procedure de resolution
        de systeme d'equations lineaires.
        """

        phi_x = self.fonction_base_polynomiale(X)
        phi_x_t = np.transpose(phi_x)

        A = (self.lamb * np.identity(self.M + 1)) + np.matmul(phi_x_t, phi_x)
        B = np.matmul(phi_x_t, t)

        return np.linalg.solve(A, B) # resoud l'equation A*X = B et retourne X

    def entrainement(self, X, t, using_sklearn=False):
        """
        Entraîne la regression lineaire sur l'ensemble d'entraînement forme des
        entrees ``X`` (un tableau 2D Numpy, ou la n-ieme rangee correspond à l'entree
        x_n) et des cibles ``t`` (un tableau 1D Numpy ou le
        n-ieme element correspond à la cible t_n). L'entraînement doit
        utiliser le poids de regularisation specifie par ``self.lamb``.

        Cette methode doit assigner le champs ``self.w`` au vecteur
        (tableau Numpy 1D) de taille D+1, tel que specifie à la section 3.1.4
        du livre de Bishop.
        
        Lorsque using_sklearn=True, vous devez utiliser la classe "Ridge" de 
        la librairie sklearn (voir http://scikit-learn.org/stable/modules/linear_model.html)
        
        Lorsque using_sklearn=Fasle, vous devez implementer l'equation 3.28 du
        livre de Bishop. Il est suggere que le calcul de ``self.w`` n'utilise
        pas d'inversion de matrice, mais utilise plutôt une procedure
        de resolution de systeme d'equations lineaires (voir np.linalg.solve).

        Aussi, la variable membre self.M sert à projeter les variables X vers un espace polynomiale de degre M
        (voir fonction self.fonction_base_polynomiale())

        NOTE IMPORTANTE : lorsque self.M <= 0, il faut trouver la bonne valeur de self.M

        """
        if self.M <= 0:
            self.recherche_hyperparametre(X, t)

        if using_sklearn:
            regression = linear_model.Ridge(self.lamb).fit(X, t)
            self.w = regression.coef_

        else:
            self.w = self.calcule_parametres_optimal(X, t)

    def prediction(self, x):
        """
        Retourne la prediction de la regression lineaire
        pour une entree, representee par un tableau 1D Numpy ``x``.

        Cette methode suppose que la methode ``entrainement()``
        a prealablement ete appelee. Elle doit utiliser le champs ``self.w``
        afin de calculer la prediction y(x,w) (equation 3.1 et 3.3).
        """

        phi_x = self.fonction_base_polynomiale(x)
        return np.matmul(self.w, phi_x[0]) # ici x est toujours un scalaire donc phi_x ne contient toujours qu'une ligne

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de la difference au carre entre
        la cible ``t`` et la prediction ``prediction``.
        """

        err = (t - prediction)**2
        return np.sum(err)
