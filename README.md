# integer JUstIfied Counterfactual Explanations (iJUICE)

Instructions to run tests with iJUICE:
If you want to test iJUICE (or any of the competitors available), do the following: 
1. Set the variable "num_instances" (line 33) equal to 20 or a number of instances desired to study [num_instances = 20]. You may also define a specific index for an instance of interest in the dataset.
2. Set the variable "datasets" equal to the list of datasets desired for running the tests. 
3. Set the variable "methods" to be a list containing "ijuice" and the name of the methods you want to test together with the iJUICE method.
4. Run main.py
5. Run plotter.py to print the obtained counterfactuals from iJUICE.
6. You may add more datasets and set their information regarding mutability, directionality and plausibility regarding the feature values.
7. To verify the justification of any CF given a model and dataset information, you may check the "verify_justification" function in the "evaluator_constructor.py" file (line 218).
