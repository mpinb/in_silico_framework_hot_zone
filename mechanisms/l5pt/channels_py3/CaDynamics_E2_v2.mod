: Dynamics that track inside calcium concentration
: modified from Destexhe et al. 1994
: extended by abast to initialize cai with minCai rather than 0 inspired by https://gist.github.com/F-A/13efddbd68469c68fab146b939c16f10 by Joram Keijser and Florian Aspart

NEURON	{
	SUFFIX CaDynamics_E2_v2
	USEION ca READ ica WRITE cai
	RANGE decay, gamma, minCai, depth
}

UNITS	{
	(mV) = (millivolt)
	(mA) = (milliamp)
	FARADAY = (faraday) (coulombs)
	(molar) = (1/liter)
	(mM) = (millimolar)
	(um)	= (micron)
}

PARAMETER	{
	gamma = 0.05 : percent of free calcium (not buffered)
	decay = 80 (ms) : rate of removal of calcium
	depth = 0.1 (um) : depth of shell
	minCai = 1e-4 (mM)
}

ASSIGNED	{ica (mA/cm2)}

STATE	{
	cai (mM)
	}
    
INITIAL{
	cai = minCai
}

BREAKPOINT	{ SOLVE states METHOD cnexp }

DERIVATIVE states	{
	cai' = -(10000)*(ica*gamma/(2*FARADAY*depth)) - (cai - minCai)/decay
}
