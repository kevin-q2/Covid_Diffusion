# Covid_Project
Data mining project to analyze the spread of COVID-19 across the US. Much of the work so far has focused on using NMF to deconstruct cumulative case counts into different patterns or waves. Graphs and detailed descriptions of our methods are written up here: [nmf report](papers/urop_report.pdf) 

### Diffusion NMF:
Our current focus is on developing a new factorization algorithm that accounts for diffusion within features of data. For this project, we want to factor COVID-19 case data to find localized "waves" and their points of origin. However our motivation for a new algorithm comes from the fact that we know that COVID-19 spreads or diffuses across locations. We assume that this diffusion can be modeled according to a [diffusion kernel](papers/diffusion_kernel.pdf), and that classic NMF algorithms can be modified to reflect that (and hence become better at finding points of origin).

[Algorithms being tested](papers/Algorithms.pdf)

### collected_data/ 
contains most of the data I am currently using. It includes clean sets of case data originally collected from Johns Hopkins: https://github.com/CSSEGISandData/COVID-19
Along with some collected census data for the purpose of normalization.  

### Analysis:

All current tests, visualizations, and analysis are made in jupyter notebooks in the analysis/ folder


