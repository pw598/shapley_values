
import streamlit as st


def run_description():

	st.title('Shapley Values')

	st.write('''Welcome! This is a short tutorial on Shapley values, and a 
	demonstration of their usage through various datasets (via the sidebar menu).''')



	st.write('''
			Shapley values help to explain how a model is working in order to better understand it, and ensure fairness, accountability, 
			and transparency in predictions. Interestingly, they are rooted in cooperative game theory, in which players form different 
			sets called coalitions, and then play and get scored on marginal contribution.

			"An intuitive way to understand the Shapley value is the following. The feature values enter the room in random order. All 
			feature values participate in the game, with equal contribution to the prediction. The Shapley value of a feature is the average 
			change in the prediction that the coalition already in the room receives when the feature value joins them."
			- Christoph Molnar, Interpretable ML (https://christophm.github.io/interpretable-ml-book/)

			It works for both classification and regression. The interpretation of the Shapley value for feature $$j$$ is: the value of the 
			$$j^{th}$$ feature contributed $$\phi_j$$ to the prediction of the particular instance compared to the average prediction in the 
			dataset.

			With a linear model:
			''')

	st.write(r'$$\hat{f}(x) = \beta_0 + \beta_1 x_1 + \ldots + \beta_p x_p$$')
	st.write(r'$$\phi_j(\hat{f}) = \beta_j x_j - E(\beta_j x_j) = \beta_j x_j - \beta_j E(X_j)$$')

	st.write('If we sum all the feature contributions for one instance, the result is:')

	st.write(r'$$\sum_{j=1}^p \phi_j (\hat{f}) = \sum_{j=1}^p (\beta_j x_j - E(\beta_j x_j))$$')
	st.write(r'$$\sum_{j=1}^p \phi_j (\hat{f}) = ( \beta_0 + \sum_{j=1}^p \beta_j x_j ) - ( \beta_0 + \sum_{j=1}^p E(\beta_j x_j) )$$')
	st.write(r'$$\sum_{j=1}^p \phi_j (\hat{f}) = \hat{f}(x) - E(\hat{f}(x))$$')

	st.write('The Shapley value is defined via a value function val of players in $$S$$. The Shapley value of a feature is its contribution to \
				the payout, weighted and summed over all possible feature value combinations.')
	
	st.write(r'$$\phi_j (val) = \sum_{S \subseteq {1, \ldots, p}\\{j}} \frac{|S|!(p - |S| - 1)!}{p!} (val(S \cup {j} - val(S)))$$')


	lst = ['$$S$$ is a subset of the features in the model', \
			'$$x$$ is the vector of feature values of the instance to be explained', \
			'$$p$$ is the number of features', \
			'$$val_x(S)$$ is the prediction for feature values in set $$S$$ that are marginalized over features that are not included in set $$S$$']

	s = ''
	for i in lst:
	    s += "- " + i + "\n"
	st.markdown(s)
