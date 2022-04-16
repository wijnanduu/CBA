# CBA
Contains code for case based reasoning, some sample datasets, and a paper submitted to HHAI2022 on the subject. 

The file 'dims.py' contains the (python) code. It lets you

* Read a csv file into a case base. The csv file is expected to be tabular data for binary classification, the label field should have the name "Label". The dimension orders will be automatically determined based on either Pearson correlation coefficients or logistic regression, depending on which option is selected. It is also possible to manually specify the orders. 
* Once the case base is created cases can be compared with each other, for instance to see if the one forces the outcome of the other. 
* Some code is available to print basic statistics about the case base, such as its consistency percentage.
