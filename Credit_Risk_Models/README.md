# Credit Risk models
This part is a readme for credit risk models and why we will choose one of the models for testing. There are 2 main models we can adopt, being gradient boosting regression and deep neural network models. Banks often refrain from assessing the credit risk of a small borrower because the small size of the loan and potentially high risk of the loan do not justify the cost of employing a professional examiner and meeting underwriting standards. The sunk cost of initiating a credit assessment is a major factor in financial exclusion of many households and small businesses. However, automated credit assessment allows FinTech companies to conduct credit scoring
of small borrowers much more frequently than traditional lending. By automating the credit rating process, FinTech credit companies could better assess the creditworthiness of borrowers through making small amounts of loans at a high frequency and monitoring the repayment behavior of the borrower. This way, FinTech credit companies can assess eligibility of borrowers for receiving loans rather than rationing those borrowers who lack sufficient standard data such as financial reports used in traditional credit risk assessment.

## Boosting
- A base model is created based on a subset of the original dataset which is used to make predictions on the whole dataset.
- Errors are calculated and observations which are incorrectly predicted, are given higher weights.
- Another model is created which tries to correct the errors from the previous model.
- Similarly, multiple models are created, each correcting the errors of the previous model.
- The final model (strong learner) is the weighted mean of all the models (weak learners).

## Deep Neural Networks
Deep neural network is simply a feedforward network with many hidden layers.  At the first layer, features are used to evaluate the value of nodes , which are then used as input for calculating nodes of the second layer. The calculation is based on a function on a weighted sum of inputs. By increasing subsequent input nodes and parameters, a deeper NN can have more  advantages compared to one layer networks (“shallow”)
• A deep network needs less neurons than a shallow one
• A shallow network is more difficult to train with our current algorithms (e.g. it
has more nasty local minima, or the convergence rate is slower)

Based on the:
• Number of layers
• Selection of activation function
• Number of perceptrons
• Normalization layers
• Dropout adjustments

we can effectively tweak these parameters to aid in our optimization process. 

