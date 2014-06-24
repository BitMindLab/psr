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

#ifndef LLNA_INFERENCE_H
#define LLNA_INFERENCE_H

#define NEWTON_THRESH 1e-10

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <stdlib.h>
#include <stdio.h>

#include "corpus.h"
#include "ctm.h"
#include "gsl-wrappers.h"

//这是一个文档的变分参数，得到的结果存入整个文集的变分参数corpus_lambda等中
typedef struct llna_var_param {

	//针对一个doc，Ulambda，Ilambda 不能变动，
	gsl_vector * Ulambda;
	gsl_vector * Ilambda;

	//在优化 Ulambda的时候，需要用到与U相关的文档，因此用到大量的Ilambda，也作为临时变量
	//gsl_vector * Jlambda; // 在采样triple中，有num_triples 个 Jlambda， 因此作为计算中的临时变量

	gsl_vector * Unu;
	gsl_vector * Inu;
	int d;  // doc_id
	int u,i, *j;
	int num_triples;


    //gsl_vector * zeta_ui;
    //gsl_vector * zeta_uij;

    gsl_matrix * phi;
    gsl_vector * phi_sum;  // 这个变量感觉也没什么用
    gsl_matrix * log_phi;



    int niter;   //一个doc（即一个样本点）的迭代次数（一般一次就可以了）
    short converged;
    double lhood;
    gsl_vector * Utopic_scores;
    gsl_vector * Vtopic_scores;
} llna_var_param;

typedef struct llna_corpus_var {
	// 全局变量，对应所有文档
    gsl_matrix* Ucorpus_lambda;
    gsl_matrix* Vcorpus_lambda;
    gsl_matrix* Ucorpus_nu;
    gsl_matrix* Vcorpus_nu;
    gsl_matrix* corpus_phi_sum;  //phi 是个tensor

    int niter;  //迭代corpus的次数


    // 局部变量，对应一个文档，计算其他参数需要用到
/*    double zeta_ui;
    double zeta_uij;
    gsl_matrix * phi; //
    gsl_matrix * log_phi;
    int niter;
    short converged;
    double lhood;*/

    //gsl_matrix* Vcorpus_phi_sum;
} llna_corpus_var;


typedef struct bundle {
    llna_var_param * var;
    llna_model * mod;
    doc * Udoc;
    doc * Vdoc;
    llna_corpus_var * c_var;
    gsl_vector * Usum_phi;
    gsl_vector * Vsum_phi;
} bundle;


/*
 * functions
 *
 */

void expected_theta(llna_var_param *var, doc* d, llna_model *mod, gsl_vector* v);

void free_temp_vectors(int size);
void free_llna_var_param(llna_var_param *);
void free_llna_Vvar_param(llna_var_param * v);
void free_llna_Uvar_param(llna_var_param * v);
double fixed_point_iter_i(int, llna_var_param *, llna_model *, doc *);

double get_zeta_ui_inv(llna_corpus_var * c_var, int u, int i);
double get_zeta_uij(llna_corpus_var * c_var, int u, int i, int j);
//double get_zeta_uij_sigmoid(llna_corpus_var * c_var, int u, int i, int j);

double sigmoid(double x);
void show_vect(gsl_vector * a, char * text);
void show_sample(int * j, int num_triples);

void  init_corpus_var(llna_corpus_var * c_var, char* start);


void init_temp_vectors(int size);
void init_Uvar_unif(llna_var_param * var, doc * Udoc, llna_model * mod);
void init_Vvar_unif(llna_var_param * var, doc * Vdoc, llna_model * mod);
void init_var(llna_corpus_var * c_var, llna_var_param * var, doc * doc, llna_model * mod);
//void init_var(llna_var_param *var, doc *doc, llna_model *mod, gsl_vector *lambda, gsl_vector *nu);

int is_contain(int * array, int size, int element);


llna_corpus_var * new_llna_corpus_var(int nusers, int nitems, int ndocs, int k);
llna_var_param * new_llna_var_param(int nterms, int k);

int update_Ulambda(llna_var_param * var, doc * Udoc, llna_model * mod, llna_corpus_var * c_var);
int opt_Vlambda(llna_var_param * var, doc * Vdoc, llna_model * mod, llna_corpus_var * c_var);
void opt_phi(llna_corpus_var * c_var, llna_var_param * var, doc * doc, llna_model * mod);



void opt_Unu(llna_corpus_var * c_var, llna_var_param * var, llna_model * mod, corpus * all_corpus);
void opt_Inu(llna_corpus_var * c_var, llna_var_param * var, llna_model * mod, corpus * all_corpus);



void opt_Unu_k(int k, llna_corpus_var * c_var, llna_var_param * var, llna_model * mod, corpus * all_corpus);
void opt_Inu_k(int k, llna_corpus_var * c_var, llna_var_param * var, llna_model * mod, corpus * all_corpus);



int opt_Uzeta(llna_var_param * var, doc * Udoc,llna_model * mod);
int opt_Vzeta(llna_var_param * var, doc * Vdoc, llna_model * mod);


void lhood_bnd(llna_corpus_var* c_var, llna_var_param * var, llna_model* mod, corpus* all_corpus);

double var_inference(llna_corpus_var * c_var, llna_var_param* var, corpus* all_corpus,
                     llna_model * mod, int d);


void update_Vexpected_ss(llna_var_param* var, doc* Vdoc, llna_corpus_var * c_var, llna_ss* ss);
void update_Uexpected_ss(llna_var_param* var, doc* Udoc, llna_corpus_var * c_var,llna_ss* ss);
void update_expected_ss(llna_corpus_var * c_var, llna_var_param* var, doc* doc, llna_ss* ss);


double sample_lhood(llna_var_param* var, doc* d, llna_model* mod);

double log_mult_prob(doc* d, gsl_vector* theta, gsl_matrix* log_beta);

void write_word_assignment(FILE* f, doc* d, gsl_matrix* phi);
void write_c_var(llna_corpus_var * c_var, char * root);

#endif
