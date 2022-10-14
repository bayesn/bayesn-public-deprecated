data {
	//Number of siblings in the galaxy
	int<lower=1> S;
	//Total number of observations
	int<lower=1> Nobs;
	//Number of observations for each supernova
	int<lower=1> Nobs_s[S];
	//Total number of interpolation points
	int<lower=1> Dint;

	//Number of knots
	int<lower=1> Nknots;
	int<lower=1> Ntknots;
	int<lower=1> Nlknots;

	//Knot values
	real tk[Nknots];
	real lk[Nknots];

	//Supernova data
	vector[Nobs] tobs;
	vector[Nobs] fobs;
	vector[Nobs] fobserr;

	//Precomputed vectors
	//Hsiao template
	vector[Dint] S0;
	//A_lambda/A_V, predefined Rv
	vector[Dint] Alam_AV;

	//CSR format precomputed matrices
	//Wavelength interpolation matrix
	int Jl_nnz;
	vector[Jl_nnz] Jl_w;
	int Jl_v[Jl_nnz];
	int Jl_u[Dint+1];
	//Passband integral
	int H_nnz;
	vector[H_nnz] H_w;
	int H_v[H_nnz];
	int H_u[Nobs+1];
	//Avs
	int UU_nnz;
	vector[UU_nnz] UU_w;
	int UU_v[UU_nnz];
	int UU_u[Dint+1];
	//delMs
	int U_nnz;
	vector[U_nnz] U_w;
	int U_v[U_nnz];
	int U_u[Nobs+1];

	//Time interpolation matrix
	matrix [Ntknots,Nobs] Jt;

	//Distance prior params
	real muhat;    //Centre prior on LCDM
	real muhaterr; //Uninformative~100mag

	//SNANA zero_point
	real ZPT;

	//SED model parameters
	matrix[Nlknots,Ntknots] W0;
	matrix[Nlknots,Ntknots] W1;
	matrix[Nknots-2*Ntknots,Nknots-2*Ntknots] L_Sigma;
	real<lower=0> tauA;
	real M0;
	real<lower=0> sigma0;

	//if 1, then choose all dMs have same value (dM_Common assumption), otherwise, iid draws of dMs (dM_Uncorrelated assumption)
	int<lower=0,upper=1> dM_correlated;
}
transformed data {
	//Convenient factor
	real gamma = log(10)/2.5;

	//Number of unpinned epsilons
	int Nfree;

	Nfree = Nknots - 2*Ntknots;
}
parameters {
	row_vector[S] theta;
	//Single distance hyperparameter
	real mu;
	//For dM_Common case, single dM parameter common to siblings (i.e. sigma_Rel=0)
	real dM_Common;
	//For dM_Uncorrelated case, S many dM parameters (choose sigma_Rel=sigma_0)
	vector[S] dM_Rel;
	vector<lower=0>[S] AV;
	vector[S*Nfree] epsilon_tform; //Stack of vectorised residuals
}
transformed parameters {
	matrix[Nfree,S] epsilon_free;
	matrix[Nlknots,Ntknots] epsilon[S];
	vector[S] delM;

	epsilon_free = L_Sigma*to_matrix(epsilon_tform, Nfree, S);
	epsilon = rep_array(rep_matrix(0, Nlknots, Ntknots), S);
	for (s in 1:S) {
		epsilon[s,2:Nlknots-1,:] = to_matrix(epsilon_free[:,s], Nlknots-2, Ntknots);
	}
	if (dM_correlated) {
		for (s in 1:S) {
			delM[s] = dM_Common;
		}
	} else {
			delM = dM_Rel;
	}
}

model {
	//Declare a variable for the latent fluxes/mags
	vector[Nobs] f;

	//Sampling statements for parameters
	theta ~ std_normal();
	mu  ~ normal(muhat, muhaterr);

	if (dM_correlated) {
		dM_Common ~ normal(0,sigma0);
	} else {
		dM_Rel ~ normal(0,sigma0);
	}
	AV ~ exponential(inv(tauA));
	epsilon_tform ~ std_normal();

	//Calculate latent flux in a scope
	{
		vector[Nlknots*Nobs] WJt;
		vector[Dint] JlWJt;
		vector[Dint] ARhost; //Host extinction at obs wavelengths
		vector[Dint] Stilde; //Host-extinguised SED
		vector[Nobs] HS;

		//Counters for tracking indices
		int counter1 = 1;
		int counter2 = 1;

		//Iterate over supernovae to do product of the knots with the time spline matrix
		for (s in 1:S) {
			//Knots for W0 + theta*W1 + epsilon
			matrix[Nlknots,Ntknots] W;
			W = W0 + theta[s]*W1 + epsilon[s];

			//Stack of vectorised W*Jt products (W interpolated to obs times)
			WJt[counter1:counter1+Nobs_s[s]*Nlknots-1] = to_vector(W*Jt[:,counter2:counter2+Nobs_s[s]-1]);
			counter1 += Nobs_s[s]*Nlknots;
			counter2 += Nobs_s[s];
		}


	//W0 + theta*W1 + epsilon interpolated to obs times and wavelengths
	JlWJt  = csr_matrix_times_vector(Dint, Nlknots*Nobs, Jl_w, Jl_v, Jl_u, WJt);

	ARhost = csr_matrix_times_vector(Dint, S, UU_w, UU_v, UU_u, AV).*Alam_AV;

	//Multiply by Hsiao template
	Stilde = S0 .* exp(-gamma*(JlWJt + ARhost));
	//Integrate through passbands
	HS = csr_matrix_times_vector(Nobs, Dint, H_w, H_v, H_u, Stilde);
	//Correct for distance and mean abs magnitude
	f = exp(gamma*(ZPT - M0 - mu - csr_matrix_times_vector(Nobs, S, U_w, U_v, U_u, delM) )) .* HS;
	}
	//Data likelihood
	fobs ~ normal(f, fobserr);
}
