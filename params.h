// (C) Copyright 2007, David M. Blei and John D. Lafferty

// This file is part of CTM-C.

// CTM-C is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// CTM-C is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#ifndef PARAMS_H
#define PARAMS_H
#define UPDATE_NU
#define MAX_MARGIN

#define DEBUG
//#define SHOW_PREDICTION
//#define PARALLEL
#define MLE 0
#define SHRINK 1
#define NUM_SAMPLE 20
#define COMMON_TOPIC 2


typedef struct llna_params
{
    int em_max_iter;
    int var_max_iter;
    int cg_max_iter;
    double em_convergence;
    double var_convergence;
    double cg_convergence;
    int cov_estimate;
    int lag;
} llna_params;

void read_params(char*);
void print_params();
void default_params();

#endif
