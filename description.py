
import streamlit as st


def run_description():

	st.title('Shapley Values')

	st.write('''
			Shapley values help to explain how a model is working in order to better understand the predictions. They are rooted 
			in cooperative game theory, in which players form different sets called coalitions, and then get scored on marginal 
			contribution.

			"The feature values enter the room in random order. All feature values participate in the game, with equal contribution 
			to the prediction. The Shapley value of a feature is the average change in the prediction that the coalition already in the 
			room receives when the feature value joins them."
			- [Interpretable ML (2022), Christoph Molnar](https://christophm.github.io/interpretable-ml-book/)


			In 1953, Lloyd Shapley derived a formula for Shapley values which satisfies four axioms:
		
			- Efficiency: the sum of contributions precisely adds up to the payout

			- Symmetry: if two players are identical, they receive equal contributions/payouts

			- Dummy (Null Player): the Shapley value for a player who doesn't contribute to any coalition is zero

			- Additivity: in a game with two value functions $v_1$ and $v_2$, the Shapley values for the sum of the games can be 
			expressed as the sum of the Shapley values.
			''')



	st.title('Marginal Contribution')

	st.write('To formalize Shapley values for the general case:')
	st.write(r'$$\phi_j = \sum_{S \subseteq N \backslash \{j\}}  \frac{ |S|!(N-|S|-1)! }{N!}  (v(S \cup \{j\}) - v(S))$$')
	st.write(' ')	
	st.write(r'- $$\sum_{S \subseteq N \backslash \{j\}}$$ is the sum over all possible coalitions without $$j$$')
	st.write(r'- $$\frac{ |S|! (N-|S|-1)! }{N!}$$ determines the weight of a marginal contribution. The $$|N|!$$ in the denominator ensures the sum of the weights is equal to $$1$$.')
	st.write(r'- $$v(S \cup \{j\}) - v(S)$$ is the core of the equation, representing the marginal contribution of player $$j$$ to coalition $$S$$')
	st.write(r'- $$1, \ldots, N$$ represents the player')
	st.write(r'- $$N$$ is the coalition of all players')
	st.write(r'- $$S$$ is a coalition')
	st.write(r'- $$|S|$$ is the size of coalition')
	st.write(r'- $$v()$$ is the value function')
	st.write(r'- $$v(N)$$ is the payout')
	st.write(r'- $$\phi_j$$ is the Shapley value')



	st.title('SHAP')

	st.write('''
			The 2017 paper ["A Unified Approach to Interpreting Model Predictions"](https://dl.acm.org/doi/10.5555/3295222.3295230)
			introduced Shapley Additive Explanations, called SHAP. They presented a way to estimate SHAP values with a linear regression and kernel function.
			Shapley values refers to the original method from game theory, SHAP is the application of Shapley values for interpreting machine learning predictions, 
			and SHAP values are the resulting values from using SHAP on the features.
			''')



	st.title('Estimating SHAP Values')

	st.write('''
			While exact Shapley values can be calculated for simple games, SHAP values must be estimated for two reasons:
			1. We only have data and lack knowledge of the true distribution.
			2. As the number of coalitions increases with the number of features ($2^n$), it may be too time-consuming to compute the marginal contributions of a feature to all coalitions.
			''')

	st.write(r'$$\hat{v}(S) = \frac{1}{n} \sum_{k=1}^n ( f(x_S^{(i)} \cup x_C^{(k)}) - f(x^{(k)}) )$$')

	st.write(r'The marginal contribution of a feature $$j$$ added to a coalition $$S$$ is given by:')

	st.write(r'$$\hat{\Delta}_{S,j} = \hat{v}(S \cup \{j\}) - \hat{v}(S)$$')
	st.write(r'$$\hat{\Delta}_{S,j} = \frac{1}{n} \sum_{k=1}^n ( f(x_{S \cup \{j\}}^{(k)} \cup x_{C \backslash \{j\}}^{(k)} - f(x_S^{(i)} \cup  x_C^{(k)}) )$$')

	st.write(r'Monte Carlo integration allows us to replace the integral with a sum and the distribution $$\mathbb{P}$$ with data samples.')



	st.title('Visualizing SHAP Values')

	st.write(r'The Python package [shap](https://shap.readthedocs.io/en/latest/index.html) provides some wonderful plotting capabilities. Use the sidebar menu to view visual results for a few different datasets, using a CatBoost model.')


	st.title('References')
	st.write('- Molnar, C. (2022). Interpretable Machine Learning: A Guide for Making Black Box Models Explainable (2nd ed.). [christophm.github.io/interpretable-ml-book/](christophm.github.io/interpretable-ml-book/)')
	st.write('- Molnar, C. Interpreting Machine Learning Models With SHAP. [https://christophmolnar.com/books/](https://christophmolnar.com/books/)')
	st.write('- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. ACM Digital Library. [https://dl.acm.org/doi/10.5555/3295222.3295230](https://dl.acm.org/doi/10.5555/3295222.3295230)')



	