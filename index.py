import streamlit as st
import pandas as pd

from penguins import run_penguins
from autompg import run_autompg
from diabetes import run_diabetes
from description import run_description

def main():

	menu = ['Description', 'Penguins', 'Auto-MPG', 'Diabetes']
	choice = st.sidebar.selectbox('Menu', menu)
	st.title(choice)

	if choice == 'Description':
		run_description()

	elif choice == 'Penguins':
		run_penguins()

	elif choice == 'Auto-MPG':
		run_autompg()

	elif choice == 'Diabetes':
		run_diabetes()


if __name__ == '__main__':
	main()
