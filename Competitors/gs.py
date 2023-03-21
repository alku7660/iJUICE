# This code has been based on Laugel et al. (2018). Full reference below:
# Laugel, T., Lesot, M. J., Marsala, C., Renard, X., & Detyniecki, M. (2017). Inverse classification for comparison-based interpretability in machine learning. arXiv preprint arXiv:1712.08443.
# Code may be found at GitHub repository:
# https://github.com/thibaultlaugel/growingspheres/blob/master/growingspheres/growingspheres.py

from Competitors.gs_support import generate_ball, generate_sphere, generate_ring
from itertools import combinations
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import time

class GS:
    """
    class to fit the Original Growing Spheres algorithm
    
    Inputs: 
    obs_to_interprete: instance whose prediction is to be interpreded
    prediction_fn: prediction function, must return an integer label
    caps: min max values of the explored area. Right now: if not None, the minimum and maximum values of the 
    target_class: target class of the CF to be generated. If None, the algorithm will look for any CF that is predicted to belong to a different class than obs_to_interprete
    n_in_layer: number of observations to generate at each step # to do
    layer_shape: shape of the layer to explore the space
    first_radius: radius of the first hyperball generated to explore the space
    decrease_radius: parameter controlling the size of the are covered at each step
    sparse: controls the sparsity of the final solution (boolean)
    verbose: text
    """
    def __init__(self,
                counterfactual,
                target_class = None,
                caps = None,
                n_in_layer = 2000,
                layer_shape = 'ring',
                first_radius = 0.1,
                decrease_radius = 10,
                sparse = True,
                verbose = False):

        self.obs_to_interprete = counterfactual.ioi.normal_x
        self.prediction_fn = counterfactual.model.model.predict
        self.y_obs = counterfactual.ioi.label      
        self.target_class = target_class
        self.caps = caps
        self.n_in_layer = n_in_layer
        self.first_radius = first_radius
        self.verbose = verbose
        self.sparse = sparse

        if decrease_radius <= 1.0:
            raise ValueError("Parameter decrease_radius must be > 1.0")
        else:
            self.decrease_radius = decrease_radius 
        if layer_shape in ['ring', 'ball', 'sphere']:
            self.layer_shape = layer_shape
        else:
            raise ValueError("Parameter layer_shape must be either 'ring', 'ball' or 'sphere'.")
        if int(self.y_obs) != self.y_obs:
            raise ValueError("Prediction function should return a class (integer)")

        self.normal_x_cf, self.run_time = self.find_counterfactual()

    def find_counterfactual(self):
        """
        Finds the decision border then perform projections to make the explanation sparse.
        """
        start_time = time.time()
        ennemies_ = self.exploration()
        closest_ennemy_ = sorted(ennemies_, 
                                key= lambda x: pairwise_distances(self.obs_to_interprete.reshape(1, -1), x.reshape(1, -1)))[0] 
        self.e_star = closest_ennemy_
        if self.sparse == True:
            out = self.feature_selection(closest_ennemy_)
        else:
            out = closest_ennemy_
        end_time = time.time()
        total_time = end_time - start_time
        return out, total_time
    
    def exploration(self):
        """
        Exploration of the feature space to find the decision boundary. Generation of instances in growing hyperspherical layers.
        """
        n_ennemies_ = 999
        radius_ = self.first_radius
        
        while n_ennemies_ > 0:
            
            first_layer_ = self.ennemies_in_layer_(radius=radius_, caps=self.caps, n=self.n_in_layer, first_layer=True)
            
            n_ennemies_ = first_layer_.shape[0]
            radius_ = radius_ / self.decrease_radius # radius gets dicreased no matter, even if no enemy?

            if self.verbose == True:
                print("%d ennemies found in initial hyperball."%n_ennemies_)
            
                if n_ennemies_ > 0:
                    print("Zooming in...")
        else:
            if self.verbose == True:
                print("Expanding hypersphere...")

            iteration = 0
            step_ = radius_ / self.decrease_radius
            #step_ = (self.decrease_radius - 1) * radius_/5.0  #To do: work on a heuristic for these parameters
            
            while n_ennemies_ <= 0:

                layer = self.ennemies_in_layer_(layer_shape=self.layer_shape, radius=radius_, step=step_, caps=self.caps,
                                                n=self.n_in_layer, first_layer=False)
                n_ennemies_ = layer.shape[0]
                radius_ = radius_ + step_
                iteration += 1

            if self.verbose == True:
                print("Final number of iterations: ", iteration)
        
        if self.verbose == True:
            print("Final radius: ", (radius_ - step_, radius_))
            print("Final number of ennemies: ", n_ennemies_)
        return layer
    
    def ennemies_in_layer_(self, layer_shape='ring', radius=None, step=None, caps=None, n=1000, first_layer=False):
        """
        Basis for GS: generates a hypersphere layer, labels it with the blackbox and returns the instances that are predicted to belong to the target class.
        """
        # todo: split generate and get_enemies

        if first_layer:
            layer = generate_ball(self.obs_to_interprete, radius, n)
        
        else:
        
            if self.layer_shape == 'ring':
                segment = (radius, radius + step)
                layer = generate_ring(self.obs_to_interprete, segment, n)
            
            elif self.layer_shape == 'sphere':
                layer = generate_sphere(self.obs_to_interprete, radius + step, n)
                
            elif self.layer_shape == 'ball':
                layer = generate_ball(self.obs_to_interprete, radius + step, n)

        #cap here: not optimal - To do
        if caps != None:
            cap_fn_ = lambda x: min(max(x, caps[0]), caps[1])
            layer = np.vectorize(cap_fn_)(layer)
            
        preds_ = self.prediction_fn(layer)
        
        if self.target_class == None:
            enemies_layer = layer[np.where(preds_ != self.y_obs)]
        else:
            enemies_layer = layer[np.where(preds_ == self.target_class)]
            
        return enemies_layer
    
    def feature_selection(self, counterfactual):
        """
        Projection step of the GS algorithm. Make projections to make (e* - obs_to_interprete) sparse. Heuristic: sort the coordinates of np.abs(e* - obs_to_interprete) in ascending order and project as long as it does not change the predicted class
        
        Inputs:
        counterfactual: e*
        """
        if self.verbose == True:
            print("Feature selection...")
            
        move_sorted = sorted(enumerate(abs(counterfactual - self.obs_to_interprete.flatten())), key=lambda x: x[1])
        move_sorted = [x[0] for x in move_sorted if x[1] > 0.0]
        
        out = counterfactual.copy()
        
        reduced = 0
        
        for k in move_sorted:
        
            new_enn = out.copy()
            new_enn[k] = self.obs_to_interprete.flatten()[k]

            if self.target_class == None:
                condition_class = self.prediction_fn(new_enn.reshape(1, -1)) != self.y_obs
                
            else:
                condition_class = self.prediction_fn(new_enn.reshape(1, -1)) == self.target_class
                
            if condition_class:
                out[k] = new_enn[k]
                reduced += 1
                
        if self.verbose == True:
            print("Reduced %d coordinates"%reduced)
        return out

    def feature_selection_all(self, counterfactual):
        """
        Try all possible combinations of projections to make the explanation as sparse as possible. 
        Warning: really long!
        """
        if self.verbose == True:
            print("Grid search for projections...")
        for k in range(self.obs_to_interprete.size):
            print('==========', k, '==========')
            for combo in combinations(range(self.obs_to_interprete.size), k):
                out = counterfactual.copy()
                new_enn = out.copy()
                for v in combo:
                    new_enn[v] = self.obs_to_interprete[v]
                if self.prediction_fn(new_enn.reshape(1, -1)) == self.target_class:
                    print('bim')
                    out = new_enn.copy()
                    reduced = k
        if self.verbose == True:
            print("Reduced %d coordinates"%reduced)
        return out