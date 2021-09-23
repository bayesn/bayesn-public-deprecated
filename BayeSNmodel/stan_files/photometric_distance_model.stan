data {
	//Total number of observations
	int<lower=1> Nobs;
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
	//A_lambda/A_V
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
	
	//Time interpolation matrix
	matrix [Ntknots,Nobs] Jt;

	//Distance prior params
	real muhat;
	real muhaterr;

	//SNANA zero_point
	real ZPT;

	//SED model parameters
	matrix[Nlknots,Ntknots] W0;
	matrix[Nlknots,Ntknots] W1;
	matrix[Nknots-2*Ntknots,Nknots-2*Ntknots] L_Sigma;
	real<lower=0> tauA;
	real M0;
	real<lower=0> sigma0;
}
transformed data {
	//Convenient factor
	real gamma = log(10)/2.5;

	//Number of unpinned epsilons
	int Nfree;

	//Standard deviation of the del_M + mu sampling variable
	real Dserr;

	Dserr = sqrt(square(muhaterr) + square(sigma0));

	Nfree = Nknots - 2*Ntknots;
}
parameters {
	real theta;
	real Ds;
	real<lower=0> AV;
	vector[Nfree] epsilon_tform; //Non-centred transform of epsilon matrix
}
transformed parameters {
	vector[Nfree] epsilon_free; //Free elements of epsilon matrix
	matrix[Nlknots,Ntknots] epsilon; //All elements of epsilon matrix

	epsilon_free = L_Sigma*epsilon_tform;
	epsilon = rep_matrix(0, Nlknots, Ntknots);
	epsilon[2:Nlknots-1, :] = to_matrix(epsilon_free, Nlknots-2, Ntknots);
}
model {
	//Declare a variable for the latent fluxes/mags
	vector[Nobs] f;
	
	//Sampling statements for parameters
	theta ~ std_normal();
	Ds ~ normal(muhat, Dserr);
	AV ~ exponential(inv(tauA));
	epsilon_tform ~ std_normal();
	
	//Calculate latent flux in a scope
	{
		vector[Nlknots*Nobs] WJt;
		vector[Dint] JlWJt;
		vector[Dint] Alam; //Host extinction at obs wavelengths
		vector[Dint] Stilde; //Host-extinguised SED
		vector[Nobs] HS;

		matrix[Nlknots,Ntknots] W; //Knots defining intrinsic SED

		//The sum of the intrinsic FCs and epsilon matrix
		W = W0 + theta*W1 + epsilon;

		//A matrix containing W*Jt in each column (reshaped into a vector)
		WJt = to_vector(W*Jt);
		
		//Interpolate intrinsic SED to obs times and wavelengths
		JlWJt = csr_matrix_times_vector(Dint, Nlknots*Nobs, Jl_w, Jl_v, Jl_u, WJt);
		Alam = Alam_AV*AV;
		
		//Multiply by Hsiao template
		Stilde = S0 .* exp(-gamma*(JlWJt + Alam));
		//Integrate through passbands
		HS = csr_matrix_times_vector(Nobs, Dint, H_w, H_v, H_u, Stilde);
		//Correct for distance and mean abs magnitude
		f = exp(gamma*(ZPT - M0 - Ds)) * HS;
	}

	//Data likelihood
	fobs ~ normal(f, fobserr);
}
generated quantities {
	real delM;
	real mu;

	//Extract mu and del_M from their linear combination
	mu = normal_rng((Ds*square(muhaterr) + muhat*square(sigma0))/square(Dserr), sqrt((square(sigma0)*square(muhaterr))/square(Dserr)));
	delM = Ds - mu;
}
