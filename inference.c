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

#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <assert.h>

#include "gsl-wrappers.h"
#include "corpus.h"
#include "ctm.h"
#include "params.h"
#include "inference.h"

extern llna_params PARAMS;

/*
 * temporary k-1 vectors so we don't have to allocate, deallocate
 *
 */
gsl_vector ** temp;  //全局变量，编译时在静态存储区分配空间，只是分配的指针空间。指针所指向内容仍然在动态存储区（堆栈）中
# if defined(PARALLEL)
#pragma omp threadprivate(temp)
# endif


void df_Ulambda(llna_corpus_var * c_var, llna_var_param * var, llna_model * mod,
		gsl_vector * df, corpus * all_corpus);





/*gsl_vector * Ulambda;
gsl_vector * Ilambda;
gsl_vector * Jlambda;
gsl_vector * Unu;
gsl_vector * Inu;
int u,i,j;*/
int ntemp = 5;

void init_temp_vectors(int size)
{
    int i;
/*    Ulambda = gsl_vector_alloc(size);
    Unu= gsl_vector_alloc(size);
    Ilambda = gsl_vector_alloc(size);
    Inu = gsl_vector_alloc(size);
    Jlambda = gsl_vector_alloc(size);*/
    temp = malloc(sizeof(gsl_vector *)*ntemp);
    for (i = 0; i < ntemp; i++)
        temp[i] = gsl_vector_alloc(size);  //temp[i]的类型是 gsl_vector *
}


void free_temp_vectors(int size)
{
/*    Ulambda = gsl_vector_alloc(size);
    Unu= gsl_vector_alloc(size);
    Ilambda = gsl_vector_alloc(size);
    Inu = gsl_vector_alloc(size);
    Jlambda = gsl_vector_alloc(size);*/
    for (int i = 0; i < ntemp; i++)
        gsl_vector_free(temp[i]);  //temp[i]的类型是 gsl_vector *
}


/**
 * check an element is NAN or not
 *
 */
int check_nan(double a, char * text)
{
	if (isnan(a))
	{
		printf("%s\n", text);
		exit(EXIT_FAILURE);
		return(1);
	}
	else
		return(0);
}

/**
 * print vector
 *
 */
void show_vect(gsl_vector * a, char * text)
{
# if defined(DEBUG)
	printf("%s", text);
	for (int i = 0; i < a->size; i++)
	{
		printf("%lf\t", vget(a, i));
	}
	printf("\n");
# endif

}


void show_sample(int * j, int num_triples)
{
# if defined(DEBUG)
	printf("triple sampling:\t");
	for (int i = 0; i < num_triples; i++)
	{
		printf("%d\t",j[i]);
	}
	printf("\n");

# endif

}





/**
 * optimize zeta
 *
 */

/*double get_zeta_ui_inv(llna_corpus_var * c_var, int u, int i)
{
    double t1;
    double zeta_ui = 0.0;

    //for (i = 0; i < mod->k-1; i++) //这里为什么是k-1，也是因为lambda(k)=0的缘故
    for (int k = 0; k < c_var->Ucorpus_lambda->size2; k++)
    {
    	t1 = 0.5 * (mget(c_var->Ucorpus_lambda, u, k) + mget(c_var->Vcorpus_lambda, i, k))
    	+ 0.125 * (mget(c_var->Ucorpus_nu, u, k) + mget(c_var->Vcorpus_nu, i, k));
    	zeta_ui += exp(t1);

    	check_nan(zeta_ui, "warning: zeta_ui is nan");
    	check_nan(1.0/zeta_ui, "warning: 1/zeta_ui is nan");
    }
	if (isinf(zeta_ui))
		return 0.000000001;
	else
		return 1.0 / zeta_ui;  //因为zeta_ui有时特别大inf，double存储不下，因此这里可以返回1/zeta_ui
	//zeta_ui如果很小，等于0,怎么办呢？

}*/

double get_zeta_ui(llna_corpus_var * c_var, int u, int i)
{
    double t1;
    double zeta_ui = 0.0;

    //for (i = 0; i < mod->k-1; i++) //这里为什么是k-1，也是因为lambda(k)=0的缘故
    for (int k = 0; k < c_var->Ucorpus_lambda->size2; k++)
    {
    	t1 = 0.5 * (mget(c_var->Ucorpus_lambda, u, k) + mget(c_var->Vcorpus_lambda, i, k))
    	+ 0.125 * (mget(c_var->Ucorpus_nu, u, k) + mget(c_var->Vcorpus_nu, i, k));
    	zeta_ui += exp(t1);

    	check_nan(zeta_ui, "warning: zeta_ui is nan");
    }
	return zeta_ui;  // zeta_ui 有可能会是inf
}


double get_zeta_uij(llna_corpus_var * c_var, int u, int i, int j)
{
    double zeta_uij = 0.0;
    for (int k = 0; k < c_var->Ucorpus_lambda->size2; k++)
    {
    	zeta_uij  += mget(c_var->Ucorpus_lambda, u, k) *
    			(mget(c_var->Vcorpus_lambda, j, k) - mget(c_var->Vcorpus_lambda, i, k));

    }
    check_nan(zeta_uij, "warning: zeta_uij is nan");
    return zeta_uij;
}

double sigmoid(double x)
{
	if (isinf(exp(x)))
		return 1.0;
	else if (isinf(exp(-x)))
		return 0.0;
	else
		return 1.0 / (1.0 + exp(-x));
}



/*
 * 如果zeta_uij太大，exp(zeta_uij)会溢出;如果太小,exp(-zeta_uij)会溢出，因此这里直接返回sigmoid(zeta_uij)比较好
 */
/*double get_zeta_uij_sigmoid(llna_corpus_var * c_var, int u, int i, int j)
{
    double zeta_uij = 0.0;
    for (int k = 0; k < c_var->Ucorpus_lambda->size2; k++)
    {
    	zeta_uij  += mget(c_var->Ucorpus_lambda, u, k) *
    			(mget(c_var->Vcorpus_lambda, j, k) - mget(c_var->Vcorpus_lambda, i, k));
    	check_nan(zeta_uij, "warning: zeta_uij is nan");

    }

	if (isinf(exp(zeta_uij)))
		return 1.0;
	else if (isinf(exp(-zeta_uij)))
		return 0.0;
	else
		return 1 / (1 + exp(-zeta_uij));
}*/




/*
 * likelihood bound
 *
 */

/*double expect_mult_norm(llna_corpus_var * c_var, llna_var_param * var)
{
    int i;
    double sum_exp = 0;
    int niter = var->Ulambda->size;
    double zeta_ui = get_zeta_ui(c_var, var->u, var->i);

    for (i = 0; i < niter; i++)
        sum_exp += exp(0.5 * (vget(var->Ulambda, i) + vget(var->Ilambda, i)) +
        		(0.125) * (vget(var->Unu,i) + vget(var->Inu,i)));

    return((zeta_ui_inv) * sum_exp - 1.0 - log(zeta_ui_inv));
}*/

double expect_mult_norm(llna_corpus_var * c_var, llna_var_param * var)
{
    /*int i;
    double sum_exp = 0;
    int niter = var->Ulambda->size;
    double zeta_ui = get_zeta_ui(c_var, var->u, var->i);

    for (i = 0; i < niter; i++)
        sum_exp += exp(0.5 * (vget(var->Ulambda, i) + vget(var->Ilambda, i)) +
        		(0.125) * (vget(var->Unu,i) + vget(var->Inu,i)));

    if(isinf(zeta_ui))
    	return (1.0 / var->Ulambda->size - 1.0 + log(zeta_ui));
    else
    	return ((1.0 / zeta_ui) * sum_exp - 1.0 + log(zeta_ui));

    if(isinf(zeta_ui))
    	return (1.0 / var->Ulambda->size);
    else*/
    return 1.0;

}


void lhood_bnd(llna_corpus_var* c_var, llna_var_param * var, llna_model* mod, corpus* all_corpus)
{

    // 1. E[log p(\eta | \mu, \Sigma)] + H(q(\eta | \lambda, \nu)
    double lhood  = (0.5) * (mod->Ulog_det_inv_cov + mod->Vlog_det_inv_cov)+ mod->k;
    for (int i = 0; i < mod->k; i++)
    {
        double v = - (0.5) * (vget(var->Unu, i) * mget(mod->Uinv_cov,i, i) +
        		vget(var->Inu, i) * mget(mod->Vinv_cov,i, i));
        for (int j = 0; j < mod->k; j++)
        {
            v -= (0.5) *
                (vget(var->Ulambda, i) - vget(mod->Umu, i)) *
                mget(mod->Uinv_cov, i, j) *
                (vget(var->Ulambda, j) - vget(mod->Umu, j));
            v -= (0.5) *
                (vget(var->Ilambda, i) - vget(mod->Vmu, i)) *
                mget(mod->Vinv_cov, i, j) *
                (vget(var->Ilambda, j) - vget(mod->Vmu, j));
        }
        v += (0.5) * (log(vget(var->Unu, i)) + log(vget(var->Inu, i)));
        lhood += v;
    }

    // 2.E[log p(R|eta1, eta2)]

# if defined(MAX_MARGIN)
    double t1, t2; int j_id;
    for (int i = 0; i < var->num_triples; i++)
    {
    	j_id = var->j[i];
    	gsl_vector Jlambda = gsl_matrix_row(c_var->Vcorpus_lambda, j_id).vector;

    	double zeta_uij = get_zeta_uij(c_var, var->u, var->i, j_id);
    	gsl_blas_ddot(var->Ulambda, var->Ilambda, &t1);
    	gsl_blas_ddot(var->Ulambda, &Jlambda, &t2);
    	printf("UI = %lf\tUJ = %lf\tzeta_uij = %lf;\t", t1, t2,zeta_uij);  //仅仅为了显示输出值
    	assert(fabs(t1 - t2 + zeta_uij) < 0.001);



        gsl_blas_dcopy(var->Ilambda, temp[2]);
        gsl_vector_sub(temp[2], &Jlambda);  //temp[2] = Ilambda-Jlambda
        gsl_blas_ddot(var->Ulambda, temp[2], &t1);
        if(t1 < 1)
        {
        	lhood += t1 - 1;
        } // else 导数0
    }
    printf("\n");

#else
    double t1, t2, t3; int j_id;
    for (int i = 0; i < var->num_triples; i++)
    {
    	j_id = var->j[i];
    	gsl_vector Jlambda = gsl_matrix_row(c_var->Vcorpus_lambda, j_id).vector;

    	double zeta_uij = get_zeta_uij(c_var, var->u, var->i, j_id);
        gsl_blas_ddot(var->Ulambda, var->Ilambda, &t2);
        gsl_blas_ddot(var->Ulambda, &Jlambda, &t3);
    	printf("UI = %lf\tUJ = %lf\tzeta_uij = %lf;\t", t2, t3,zeta_uij);  //仅仅为了显示输出值
    	assert(fabs(t1 - t2 + zeta_uij) < 0.001);

        t1 =sigmoid(zeta_uij);




        lhood += log(1 - t1) - t1 * (t3 - t2 - zeta_uij);
    }
    printf("\n");
#endif

    // 3.E[log p(z_n | \eta)] + E[log p(w_n | \beta)] + H(q(z_n | \phi_n))
    // 这里只考虑一个文档，在total= 所有lhood的累加，得到所有的lhood
    // 这部分还没写完。。。。。。。


    doc doc = all_corpus->docs[var->d];
    int total = doc.total;
    lhood -= expect_mult_norm(c_var, var) * total;

    for (int i = 0; i < doc.nterms; i++)
    {
        // !!! we can speed this up by turning it into a dot product
        // !!! profiler says this is where some time is spent
        for (int j = 0; j < mod->k; j++)
        {
        	if (i >= var->phi->size1 || j >= var->phi->size2)
        	{
        		printf("warning: out of range in lhood_bnd 2");
        	}

            double phi_ij = mget(var->phi, i, j);
            double log_phi_ij = mget(var->log_phi, i, j);



            if (phi_ij > 0)
            {
                vinc(var->Utopic_scores, j, phi_ij * doc.count[i]);  // Utopic_scores干嘛用的？
                lhood +=
                    doc.count[i] * phi_ij *
                    (vget(var->Ulambda, j) +
                     mget(mod->log_beta, j, doc.word[i]) -
                     log_phi_ij);
            }
        }
    }
    var->lhood = lhood;
}



/**
 * detect is contain
 *
 */

int is_contain(int * array, int size, int element)
{
	int flag = 0;

	for (int i = 0; i < size; i++)
	{
		if (array[i] == element)
		{
			flag = 1;
			break;
		}
	}
	return flag;
}


/**
 * Sample a triple for SGD
 *
 */

void SampleTriple(corpus* all_corpus, llna_var_param * var)
{
	int v_id;
	vect umatrix = all_corpus->usermatrix[var->u];  //
	int i;
	for (i = 0; i < var->num_triples; i++)
	{
		do
		{
			v_id = rand()%all_corpus->nitem;             //随机选一个不喜欢的item
		}
		while (is_contain(umatrix.id, umatrix.size, v_id) == 1 ||
				is_contain(var->j, var->num_triples, v_id));
		// 采样到喜欢的，需要重新采样;采样到已经采样过的，也需要重新采样
		var->j[i] = v_id;

	}
	if (i != var->num_triples)
		printf("warning: %d triple is oversampled \n", var->num_triples);
}





/*void opt_zeta_uij(llna_corpus_var * c_var, llna_var_param* var, llna_model * mod)
{
    double t1, t2;
    var->zeta_uij = 0.0;

    for (int i = 0; i < var->num_triples; i++)
    {
    	var->Jlambda = gsl_matrix_row(c_var->Vcorpus_lambda, var->j[i]).vector;
    	gsl_blas_ddot(var->Ulambda, var->Ilambda, &t1);
    	gsl_blas_ddot(var->Ulambda, var->Jlambda, &t2);
    	var->zeta_uij[i] = t2 - t1;
    }
}*/


/**
 * optimize phi
 *
 */

void opt_phi(llna_corpus_var * c_var, llna_var_param * var, doc * doc, llna_model * mod)
{
    int i, n, K = mod->k;
    double log_sum_n = 0;

    // compute phi proportions in log space

    gsl_vector * sum_phi = gsl_vector_alloc(mod->k);
    gsl_vector_set_zero(sum_phi);
    for (n = 0; n < doc->nterms; n++)
    {
    	// 更新 局部变量 var->phi
        log_sum_n = 0;
        for (i = 0; i < K; i++)
        {
            mset(var->log_phi, n, i,
                 0.5 * (mget(c_var->Ucorpus_lambda, doc->u_id, i) + mget(c_var->Vcorpus_lambda, doc->v_id, i))
                 + mget(mod->log_beta, i, doc->word[n]));
            if (i == 0)
                log_sum_n = mget(var->log_phi, n, i);
            else
                log_sum_n =  log_sum(log_sum_n, mget(var->log_phi, n, i));
        }
        for (i = 0; i < K; i++)
        {
            mset(var->log_phi, n, i, mget(var->log_phi, n, i) - log_sum_n);  //???
            mset(var->phi, n, i, exp(mget(var->log_phi, n, i)));
        }

        // 更新 全局变量  var->corpus_phi_sum
        for (i = 0; i < mod->k; i++)
        {
            vset(sum_phi, i,
                 vget(sum_phi, i) +
                 ((double) doc->count[n]) * mget(var->phi, n, i)); //sum_phi-----13
        }
    }
    show_vect(sum_phi, "sum_phi = ");
    gsl_matrix_set_row(c_var->corpus_phi_sum, doc->d_id, sum_phi);
    gsl_vector_free(sum_phi);

}

/**
 * optimize lambda
 *
 */

/*void fdf_Ulambda(const gsl_vector * p, void * params, double * f, gsl_vector * df)
{
    *f = f_Ulambda(p, params);
    df_Ulambda(p, params, df);
}
void fdf_Vlambda(const gsl_vector * p, void * params, double * f, gsl_vector * df)
{
    *f = f_Vlambda(p, params);
    df_Vlambda(p, params, df);
}*/

/*
double f_Ulambda(const gsl_vector * p, void * params)  //目标函数中所有带 lambda的项, p 是变量,即lambda
{
    //return(-(term1+term2+term3));
}
double f_Vlambda(const gsl_vector * p, void * params)  //目标函数中所有带 lambda的项, p 是变量,即lambda
{
}
*/

// 自变量p
//this function should store the n-dimensional gradient df_i = d f(p,params) / d x_i
void df_Ulambda(llna_corpus_var * c_var, llna_var_param * var, llna_model * mod,
		gsl_vector * df, corpus * all_corpus)
{

    //1. compute \Sigma^{-1} (\mu - \lambda) = temp[0]
    gsl_vector_set_zero(temp[0]);
    gsl_blas_dcopy(mod->Umu, temp[1]);
    gsl_vector_sub(temp[1], var->Ulambda);  //temp[1]=mu-lambda
    gsl_blas_dsymv(CblasLower, 1, mod->Uinv_cov, temp[1], 0, temp[0]); //temp[0]=inv*temp[1]

    for (int i = 0; i < mod->k; i++)
    	check_nan(vget(temp[0], i), "warning: dUlambda--0\n");



    //2. compute zeta_uij / (1 + zeta_uij) * (Ilambda - Jlambda) = temp[1]
# if defined(MAX_MARGIN)
    gsl_vector_set_zero(temp[1]);
    double t1; int j_id;
    for (int i = 0; i < var->num_triples; i++)
    {
    	j_id = var->j[i];
    	gsl_vector Jlambda = gsl_matrix_row(c_var->Vcorpus_lambda, j_id).vector;
    	gsl_blas_dcopy(var->Ilambda, temp[2]);
        gsl_vector_sub(temp[2], &Jlambda);  //temp[2] = Ilambda-Jlambda
        gsl_blas_ddot(var->Ulambda, temp[2], &t1);
        if(t1 < 1)
        {
        	gsl_vector_add(temp[1],temp[2]);
        } // else 导数0
    }

#else
    gsl_vector_set_zero(temp[1]);
    double t1; int j_id;
    for (int i = 0; i < var->num_triples; i++)
    {
    	j_id = var->j[i];
    	double zeta_uij = get_zeta_uij(c_var, var->u, var->i, j_id);

    	//zeta_uij = -700.2;

    	gsl_vector Jlambda = gsl_matrix_row(c_var->Vcorpus_lambda, j_id).vector;
        t1 =sigmoid(zeta_uij);

        gsl_blas_dcopy(var->Ilambda, temp[2]);
        gsl_vector_sub(temp[2], &Jlambda);  //temp[2] = Ilambda-Jlambda
        gsl_vector_scale(temp[2], t1);   //temp[2] = t1*(Ilambda-Jlambda)

        gsl_vector_add(temp[1],temp[2]);
    }

#endif


    for (int i = 0; i < mod->k; i++)
    	check_nan(vget(temp[1], i), "warning: dUlambda--1\n");

    //3. compute  sum( sum_phi )   temp[2]
    vect udoc_list = all_corpus->udoc_list[var->u];
    gsl_vector_set_zero(temp[2]);
    for (int n = 0; n < udoc_list.size; n++)
    {
    	int doc_id = udoc_list.id[n];
    	gsl_vector tt = gsl_matrix_row(c_var->corpus_phi_sum, doc_id).vector;// .....以上都已经检查无误！

    	gsl_vector_add(temp[2], &tt);
    }
    gsl_vector_scale(temp[2], 0.5); // temp[2]


    for (int i = 0; i < mod->k; i++)
    	check_nan(vget(temp[2], i), "warning: dUlambda--2\n");

    //3. compute - (N / \zeta) * exp(\lambda + \nu^2 / 2) = temp[3]
    gsl_vector_set_zero(temp[3]);
    for (int n = 0; n < udoc_list.size; n++)
    {
    	int doc_id = udoc_list.id[n];
    	int doc_total = all_corpus->docs[doc_id].total;
    	int v_id = all_corpus->docs[doc_id].v_id;
    	double zeta_ui = get_zeta_ui(c_var, var->u, v_id);

        for (int i = 0; i < mod->k; i++)
        {
        	double tt = exp(0.5 * (vget(var->Ulambda, i) + mget(c_var->Vcorpus_lambda, v_id, i))+
    				0.125 * (vget(var->Unu, i) + mget(c_var->Vcorpus_nu, v_id, i)));
        	if(isinf(zeta_ui) || isinf(tt) || fabs(zeta_ui) < 0.00001)
        		tt = 1.0 / mod->k;
        	else
        		tt = tt / zeta_ui;

            vset(temp[4], i, -(double) doc_total * tt);

            check_nan(vget(temp[4], i), "warning: dUlambda--3\n");
        }
        gsl_vector_add(temp[3], temp[4]);
        gsl_vector_scale(temp[3], 0.5);
    }



    // set return value (note negating derivative of bound)

    gsl_vector_set_all(df, 0.0);
    gsl_vector_add(df, temp[0]); //df = df-temp[0]
    gsl_vector_add(df, temp[1]);
    gsl_vector_add(df, temp[2]);
    gsl_vector_add(df, temp[3]);


# if defined(DEBUG)
    show_vect(temp[0], "temp0=");
    show_vect(temp[1], "temp1=");
    show_vect(temp[2], "temp2=");
    show_vect(temp[3], "temp3=");
    show_vect(df, "df_Ulambda=");
# endif
}

void df_Ilambda(llna_corpus_var * c_var, llna_var_param * var, llna_model * mod,
		gsl_vector * df, corpus * all_corpus)
{

    //1. compute \Sigma^{-1} (\mu - \lambda) = temp[0]
    gsl_vector_set_zero(temp[0]);
    gsl_blas_dcopy(mod->Vmu, temp[1]);
    gsl_vector_sub(temp[1], var->Ilambda);  //temp[1]=mu-lambda
    gsl_blas_dsymv(CblasLower, 1, mod->Vinv_cov, temp[1], 0, temp[0]); //temp[0]=inv*temp[1]

    for (int i = 0; i < mod->k; i++)
    	check_nan(vget(temp[0], i), "warning: dIlambda--0\n");


    //2. compute temp[1]
# if defined(MAX_MARGIN)
    gsl_vector_set_zero(temp[1]);
    double t1; int j_id;
    for (int i = 0; i < var->num_triples; i++)
    {
        j_id = var->j[i];
        gsl_vector Jlambda = gsl_matrix_row(c_var->Vcorpus_lambda, j_id).vector;
        gsl_blas_dcopy(var->Ilambda, temp[2]);
        gsl_vector_sub(temp[2], &Jlambda);  //temp[2] = Ilambda-Jlambda
        gsl_blas_ddot(var->Ulambda, temp[2], &t1);

        if(t1 < 1)
        {
            gsl_vector_add(temp[1], var->Ilambda);
        } // else 导数0
    }

#else
    gsl_vector_set_zero(temp[1]);
    double t1; int j_id;
    for (int i = 0; i < var->num_triples; i++)
    {
    	j_id = var->j[i];
    	double zeta_uij = get_zeta_uij(c_var, var->u, var->i, j_id);

    	//gsl_vector Jlambda = gsl_matrix_row(c_var->Vcorpus_lambda, j_id).vector;
        t1 =sigmoid(zeta_uij);

        gsl_blas_dcopy(var->Ulambda, temp[2]);  //temp[2] = Ulambda
        gsl_vector_scale(temp[2], t1);   //temp[2] = t1* Ulambda

        gsl_vector_add(temp[1],temp[2]);
        //exit();
        //goto error_end;
    }
#endif

    for (int i = 0; i < mod->k; i++)
    	check_nan(vget(temp[1], i), "warning: dIlambda error in part 1\n");

    //3. compute  sum_phi   temp[2]
    vect idoc_list = all_corpus->idoc_list[var->i];
    gsl_vector_set_zero(temp[2]);
    for (int n = 0; n < idoc_list.size; n++)
    {
    	int doc_id = idoc_list.id[n];
    	gsl_vector tt = gsl_matrix_row(c_var->corpus_phi_sum, doc_id).vector;

    	/*double aaa = 0.0;
    	for(int k = 0; k < mod->k; k++ )
    		aaa += vget(&tt, k);
    	printf("is_equal 1: %lf\n", aaa);*/

    	gsl_vector_add(temp[2], &tt);
    }
    gsl_vector_scale(temp[2], 0.5); // temp[2]

    for (int i = 0; i < mod->k; i++)
    	check_nan(vget(temp[2], i), "warning: dIlambda error in part 2\n");

    //3. compute - (N / \zeta) * exp(\lambda + \nu^2 / 2) = temp[3]
    gsl_vector_set_zero(temp[3]);
    for (int n = 0; n < idoc_list.size; n++)
    {
    	int doc_id = idoc_list.id[n];
    	int doc_total = all_corpus->docs[doc_id].total;
    	int u_id = all_corpus->docs[doc_id].u_id;
    	double zeta_ui = get_zeta_ui(c_var, u_id, var->i);

        for (int i = 0; i < mod->k; i++)
        {
        	double tt = exp(0.5 * (mget(c_var->Ucorpus_lambda, u_id, i) + vget(var->Ilambda, i)) +
    				0.125 * ( mget(c_var->Ucorpus_nu, u_id, i) + vget(var->Inu, i)));
        	if(isinf(zeta_ui) || isinf(tt) || fabs(zeta_ui) < 0.00001)
        		tt = 1.0 / mod->k;
        	else
        		tt = tt / zeta_ui;

            vset(temp[4], i, -(double) doc_total * tt);
            check_nan(vget(temp[4], i), "warning: dIlambda--3\n");
        }
        gsl_vector_add(temp[3], temp[4]);
        gsl_vector_scale(temp[3], 0.5);
    }


    // set return value (note negating derivative of bound)

    gsl_vector_set_all(df, 0.0);
    gsl_vector_add(df, temp[0]); //df = df-temp[0]
    gsl_vector_add(df, temp[1]);
    gsl_vector_add(df, temp[2]);
    gsl_vector_add(df, temp[3]);


# if defined(DEBUG)
    show_vect(temp[0], "temp0=");
    show_vect(temp[1], "temp1=");
    show_vect(temp[2], "temp2=");
    show_vect(temp[3], "temp3=");
    show_vect(df, "df_Ilambda=");
# endif

}

void df_Jlambda(llna_corpus_var * c_var, llna_var_param * var, llna_model * mod,
		gsl_vector * df, corpus * all_corpus)
{
}




/**
 * optimize nu
 *
 */

/*double f_nu_i(double nu_i, int i, llna_var_param * var,
              llna_model * mod, doc * d)
{
    double v;

    v = - (nu_i * mget(mod->inv_cov, i, i) * 0.5)
        - (((double) d->total/var->zeta) * exp(vget(var->lambda, i) + nu_i/2))
        + (0.5 * safe_log(nu_i));

    return(0);
}*/

//自变量nu_i
double df_Unu_k(double nu_k, int k, llna_corpus_var * c_var, llna_var_param * var,
               llna_model * mod, corpus * all_corpus)
{
    double v =0.0;

    vect udoc_list = all_corpus->udoc_list[var->u];

    for (int n = 0; n < udoc_list.size; n++)
    {
    	int doc_id = udoc_list.id[n];
    	int doc_total = all_corpus->docs[doc_id].total;
    	int v_id = all_corpus->docs[doc_id].v_id;
    	double zeta_ui = get_zeta_ui(c_var, var->u, v_id);

        double tt = exp(0.5 * (vget(var->Ulambda, k) + mget(c_var->Vcorpus_lambda, v_id, k))+
        		0.125 * (nu_k + mget(c_var->Vcorpus_nu, v_id, k)));

    	if(isinf(zeta_ui) || isinf(tt) || fabs(zeta_ui) < 0.00001)
    		tt = 1.0 / mod->k;
    	else
    	{
            tt = tt / zeta_ui;
    	}

    	v -= 0.125 * (double) doc_total * tt;

        if(isinf(v))
        {
    		printf("warning: v is inf\n");
    		exit(EXIT_FAILURE);
        }
    }


    v += -mget(mod->Uinv_cov, k, k) * 0.5 + 0.5 / nu_k;

    return(v);
}


double df_Inu_k(double nu_k, int k, llna_corpus_var * c_var, llna_var_param * var,
               llna_model * mod, corpus * all_corpus)
{

    double v =0.0;

    vect idoc_list = all_corpus->idoc_list[var->i];

    for (int n = 0; n < idoc_list.size; n++)
    {
    	int doc_id = idoc_list.id[n];
    	int doc_total = all_corpus->docs[doc_id].total;
    	int u_id = all_corpus->docs[doc_id].u_id;
    	double zeta_ui = get_zeta_ui(c_var, u_id, var->i);

    	double tt = exp(0.5 * (mget(c_var->Ucorpus_lambda, u_id, k) + vget(var->Ilambda, k))+
    			0.125 * (mget(c_var->Ucorpus_nu, u_id, k) + nu_k));
    	if(isinf(zeta_ui) || isinf(tt) || fabs(zeta_ui) < 0.00001)
    		tt = 1.0 / mod->k;
    	else
    	{
    		tt = tt / zeta_ui;
    	}

    	v -= 0.125 * (double) doc_total * tt;

        if(isinf(v))
        {
    		printf("warning: v is inf\n");
    		exit(EXIT_FAILURE);
        }
    	check_nan(v, "warning: df_Inu_k is nan");
    }


    v += -mget(mod->Vinv_cov, k, k) * 0.5 + 0.5 / nu_k;

    return(v);
}


double d2f_Unu_k(double nu_k, int k, llna_corpus_var * c_var, llna_var_param * var, llna_model * mod, corpus * all_corpus)
{
    double v = 0.0;

    vect udoc_list = all_corpus->udoc_list[var->u];
    for (int n = 0; n < udoc_list.size; n++)
    {
    	int doc_id = udoc_list.id[n];
    	int doc_total = all_corpus->docs[doc_id].total;
    	int v_id = all_corpus->docs[doc_id].v_id;
    	double zeta_ui = get_zeta_ui(c_var, var->u, v_id);

    	double tt = exp(0.5 * (vget(var->Ulambda, k) + mget(c_var->Vcorpus_lambda, v_id, k))+
    			0.125 * (nu_k + mget(c_var->Vcorpus_nu, v_id, k)));
    	if(isinf(zeta_ui) || isinf(tt) || fabs(zeta_ui) < 0.00001)
    		tt = 1.0 / mod->k;
    	else
    		tt = tt / zeta_ui;

        v += - 0.125 * 0.125 * (double) doc_total * tt;

        if(isinf(v))
        {
    		printf("warning: v is inf\n");
    		exit(EXIT_FAILURE);
        }

    }
    v -= 0.5 / (nu_k * nu_k);

    return(v);
}

double d2f_Inu_k(double nu_k, int k, llna_corpus_var * c_var, llna_var_param * var, llna_model * mod, corpus * all_corpus)
{
    double v = 0.0;

    vect idoc_list = all_corpus->idoc_list[var->i];
    for (int n = 0; n < idoc_list.size; n++)
    {
    	int doc_id = idoc_list.id[n];
    	int doc_total = all_corpus->docs[doc_id].total;
    	int u_id = all_corpus->docs[doc_id].u_id;
    	double zeta_ui = get_zeta_ui(c_var, u_id, var->i);

    	double tt = exp(0.5 * (mget(c_var->Ucorpus_lambda, u_id, k) + vget(var->Ilambda, k)) +
	       		 0.125 * (mget(c_var->Ucorpus_nu, u_id, k) + nu_k));
    	if(isinf(zeta_ui) || isinf(tt) || fabs(zeta_ui) < 0.00001)
    		tt = 1.0 / mod->k;
    	else
    		tt = tt / zeta_ui;

        v += - 0.125 * 0.125 * (double) doc_total * tt;

        if(isinf(v))
        {
    		printf("warning: v is inf\n");
    		exit(EXIT_FAILURE);
        }

    }
    v -= 0.5 / (nu_k * nu_k);

    return(v);
}

void opt_Unu(llna_corpus_var * c_var, llna_var_param * var, llna_model * mod, corpus * all_corpus)
{
    int k;

    // !!! here i changed to k-1
    for (k = 0; k < mod->k; k++)
    	opt_Unu_k(k, c_var, var, mod, all_corpus);

}
void opt_Inu(llna_corpus_var * c_var, llna_var_param * var, llna_model * mod, corpus * all_corpus)
{
    int k;

    // !!! here i changed to k-1
    for (k = 0; k < mod->k; k++)
    	opt_Inu_k(k, c_var, var, mod, all_corpus);
}



void opt_Unu_k(int k, llna_corpus_var * c_var, llna_var_param * var, llna_model * mod, corpus * all_corpus)
{
    double init_nu = 1.0;
    double nu_k = 0, log_nu_k = 0, df = 0, d2f = 0;
    int iter = 0;

    log_nu_k = log(init_nu);
    do
    {
        iter++;

		nu_k = exp(log_nu_k);

		// 方差太大，就没什么意义啦吧，相当于没有这个约束了。
		// 另外nu_k太大，
		// 由于zeta_ui的存在，df不会绝对值很大
        if (isnan(nu_k) || nu_k > 10 || nu_k < 0.0001)
        {
            init_nu = 0.5;
            printf("warning : nu is nan; new init = %5.5f\n", init_nu);
            log_nu_k = log(init_nu);
            nu_k = init_nu;
        }
        // f = f_nu_i(nu_i, i, var, mod, d);
        // printf("%5.5f  %5.5f \n", nu_i, f);
        df = df_Unu_k(nu_k, k, c_var, var, mod, all_corpus);
        d2f = d2f_Unu_k(nu_k, k, c_var, var, mod, all_corpus);  // 这里不能是inf

        if(isinf(df) || isinf(d2f))
        {
    		printf("warning: df or d2f is inf\n");
    		exit(EXIT_FAILURE);
        }

        printf("df = %lf, d2f=%lf, nu_k=%lf, log_nu_k=%lf\n", df, d2f, nu_k, log_nu_k);

        check_nan(df, "warning: Unu-df is nan");
        check_nan(d2f, "warning: Unu-d2f is nan");

        // 选择性更新，优先采用newton法更新，newton不适合，才采用最速梯度法更新
        if(fabs(d2f*nu_k*nu_k+df*nu_k) > 0.0001)
        	log_nu_k = log_nu_k - (df * nu_k) / (d2f * nu_k * nu_k + df * nu_k);
        else
        	log_nu_k = log_nu_k + 0.03 * df * nu_k;

        check_nan(log_nu_k, "warning: Unu-log_nu_k is nan");
        if (isinf(exp(log_nu_k)))
        	log_nu_k = log(0.5);  //等价与nu_k = 0.5


        //check_nan(exp(log_nu_k), "warning: Unu-exp(log_nu_k) is nan");
    }
    while (fabs(df) > NEWTON_THRESH && iter < 100);

    // 鉴于合理性，保证nu不会发散，以至于特别大，这里强制设定nu《10.
    // 但是不能解决根本问题，根本问题是 本不应该发散的
    double tt = exp(log_nu_k);
    if (tt < 10 && tt > 0.0001)
    	vset(var->Unu, k, tt); // else 就不更新

}

void opt_Inu_k(int k, llna_corpus_var * c_var, llna_var_param * var, llna_model * mod, corpus * all_corpus)
{

    double init_nu = 1.0;
    double nu_k = 0, log_nu_k = 0, df = 0, d2f = 0;
    int iter = 0;

    log_nu_k = log(init_nu);
    do
    {
        iter++;
        nu_k = exp(log_nu_k);
        // assert(!isnan(nu_i));
        if (isnan(nu_k) || nu_k > 10 || nu_k < 0.0001)
        {
            init_nu = 0.5;
            printf("warning : nu is nan; new init = %5.5f\n", init_nu);
            log_nu_k = log(init_nu);
            nu_k = init_nu;
        }
        // f = f_nu_i(nu_i, i, var, mod, d);
        // printf("%5.5f  %5.5f \n", nu_i, f);
        df = df_Inu_k(nu_k, k, c_var, var, mod, all_corpus);
        d2f = d2f_Inu_k(nu_k, k, c_var, var, mod, all_corpus);  // 这里df和d2f可能是inf或-inf

        if(isinf(df) || isinf(d2f))
        {
    		printf("warning: df or d2f is inf\n");
    		exit(EXIT_FAILURE);
        }

        printf("df = %lf, d2f=%lf, nu_k=%lf, log_nu_k=%lf\n", df, d2f, nu_k, log_nu_k);

        check_nan(df, "warning: Inu-df is nan");
        check_nan(d2f, "warning: Inu-d2f is nan");

        // 选择性更新，优先采用newton法更新，newton不适合，才采用最速梯度法更新
        if(fabs((d2f * nu_k * nu_k + df * nu_k)) > 0.0001)
        	log_nu_k = log_nu_k - (df * nu_k) / (d2f * nu_k * nu_k + df * nu_k);
        else
        	log_nu_k = log_nu_k + 0.03 * df * nu_k;

        check_nan(log_nu_k, "warning: Inu-log_nu_k is nan");
        if (isinf(exp(log_nu_k)))
        	log_nu_k = log(0.5);  //等价与nu_k = 0.5

    }
    while (fabs(df) > NEWTON_THRESH && iter < 100);

    double tt = exp(log_nu_k);
    if (tt < 10 && tt > 0.0001)
        vset(var->Inu, k, tt); // else 就不更新
}



/**
 * initial variational parameters
 *
 */

void init_var_unif(llna_var_param * var, doc * Udoc, llna_model * mod)
{
    int i;

    gsl_matrix_set_all(var->phi, 1.0/mod->k);
    gsl_matrix_set_all(var->log_phi, -log((double) mod->k));

    //for (i = 0; i < mod->k-1; i++)
    for (i = 0; i < mod->k; i++)
    {
        //vset(var->nu, i, 10.0);
        vset(var->Ulambda, i, 1.0);
        vset(var->Unu, i, 1.0);
    }
    var->niter = 0;
    var->lhood = 0;
}


void init_var( llna_corpus_var * c_var, llna_var_param * var, doc * doc, llna_model * mod)
{
	var->d = doc->d_id;
	var->u = doc->u_id;
	var->i = doc->v_id;
	gsl_vector Ulambda = gsl_matrix_row(c_var->Ucorpus_lambda, var->u).vector;
	gsl_vector Ilambda = gsl_matrix_row(c_var->Vcorpus_lambda, var->i).vector;
	gsl_vector Unu = gsl_matrix_row(c_var->Ucorpus_nu, var->u).vector;
	gsl_vector Inu = gsl_matrix_row(c_var->Vcorpus_nu, var->i).vector;

	gsl_blas_dcopy(&Ulambda, var->Ulambda);
	gsl_blas_dcopy(&Ilambda, var->Ilambda);
	gsl_blas_dcopy(&Unu, var->Unu);
	gsl_blas_dcopy(&Inu, var->Inu);



    //get_zeta_ui(c_var, var->u, var->i);
    //get_zeta_uij(c_var, c_var, mod);
    opt_phi(c_var, var, doc, mod);  // 这里是指针

    var->niter = 1;
    var->num_triples = NUM_SAMPLE;
    var->lhood = 0.0;

    var->j =  malloc(sizeof(int) * var->num_triples);
    for (int i = 0; i < var->num_triples; i++)
    	var->j[i] = -1;  // means not legal value

}
/*
void init_Vvar(llna_var_param * var, doc * Vdoc, llna_model * mod, gsl_vector *Vlambda, gsl_vector* Vnu)
{
    gsl_vector_memcpy(var->Vlambda, Vlambda);
    gsl_vector_memcpy(var->Vnu, Vnu);

    opt_Vzeta(var, Vdoc, mod);
    opt_Vphi(var, Vdoc, mod);
    var->niter = 0;
}
*/




/**
 *
 * variational inference
 *
 */

llna_var_param * new_llna_var_param(int nterms, int k)
{
    llna_var_param * ret = malloc(sizeof(llna_var_param));
    ret->Ulambda = gsl_vector_alloc(k);
    ret->Ilambda = gsl_vector_alloc(k);
    ret->Unu = gsl_vector_alloc(k);
    ret->Inu = gsl_vector_alloc(k);

    //ret->Unu = malloc(sizeof(double)); //因为不是指针，因此就没必要申请空间
    ret->phi = gsl_matrix_alloc(nterms, k);
    ret->log_phi = gsl_matrix_alloc(nterms, k);
    ret->phi_sum = gsl_vector_alloc(k);


    ret->Utopic_scores = gsl_vector_alloc(k); //???干嘛的
    ret->Vtopic_scores = gsl_vector_alloc(k);
    return(ret);
}

/*
llna_var_param * new_llna_Vvar_param(int Vnterms, int k)
{
    llna_var_param * ret = malloc(sizeof(llna_var_param));
    ret->Vlambda = gsl_vector_alloc(k);
    ret->Vnu = gsl_vector_alloc(k);
    ret->Vphi = gsl_matrix_alloc(Vnterms, k);
    ret->Vlog_phi = gsl_matrix_alloc(Vnterms, k);
    ret->Vzeta = 0;
    ret->Vtopic_scores = gsl_vector_alloc(k); //???干嘛的
    return(ret);
}
*/
llna_corpus_var * new_llna_corpus_var(int nusers, int nitems, int ndocs, int k)
{
	llna_corpus_var * c_var = malloc(sizeof(llna_corpus_var));
    c_var->Ucorpus_lambda = gsl_matrix_alloc(nusers, k);
    c_var->Vcorpus_lambda = gsl_matrix_alloc(nitems, k);

    c_var->Ucorpus_nu = gsl_matrix_alloc(nusers, k);
    c_var->Vcorpus_nu = gsl_matrix_alloc(nitems, k);

    c_var->corpus_phi_sum = gsl_matrix_alloc(ndocs, k);  //这里竟然也0初始化了？
    //c_var->Vcorpus_phi_sum = gsl_matrix_alloc(nitems, k);
    return(c_var);
}
void  init_corpus_var(llna_corpus_var * c_var, char* start)
{
	if (strcmp(start, "rand")==0) {

		gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);   //用陶斯沃特方法 ，定义一个随机数生成器r
	    long t1;
	    double val;


	    (void) time(&t1);
	    gsl_rng_set(r, t1); //seed the random number generator,如果两次的种子相同，那么生成的值也相同


	    for (int i = 0; i < c_var->Ucorpus_lambda->size1; i++)
	    {
	    	for (int j = 0; j < c_var->Ucorpus_lambda->size2; j++)
	    	{
	    		val = gsl_rng_uniform(r) + 1.0/100;   //gsl_rng_uniform生成[0,1)随机数,后者估计是用来平滑的
	    		mset(c_var->Ucorpus_lambda, i, j, val);
	    	}
	    }

	    for (int i = 0; i < c_var->Vcorpus_lambda->size1; i++)
	    {
	    	for (int j = 0; j < c_var->Vcorpus_lambda->size2; j++)
	    	{
	    		val = gsl_rng_uniform(r) + 1.0/100;   //gsl_rng_uniform生成[0,1)随机数,后者估计是用来平滑的
	    		mset(c_var->Vcorpus_lambda, i, j, val);
	    	}
	    }

		gsl_matrix_set_all(c_var->Ucorpus_nu, 1);
		gsl_matrix_set_all(c_var->Vcorpus_nu, 1);
		gsl_matrix_set_all(c_var->corpus_phi_sum, 0.001); // phi_sum不需要满足加和为1，可以设置为0,或者一个小的数字作为平滑


		gsl_rng_free(r);
	} else {
		char fname[100];
		sprintf(fname, "%s-Ucorpus_lambda.dat", start);
		scanf_matrix(fname, c_var->Ucorpus_lambda);
        sprintf(fname, "%s-Vcorpus_lambda.dat", start);
        scanf_matrix(fname, c_var->Vcorpus_lambda);

		sprintf(fname, "%s-Ucorpus_nu.dat", start);
		scanf_matrix(fname, c_var->Ucorpus_nu);
        sprintf(fname, "%s-Vcorpus_nu.dat", start);
        scanf_matrix(fname, c_var->Vcorpus_nu);

        sprintf(fname, "%s-corpus_phi_sum.dat", start);
        scanf_matrix(fname, c_var->corpus_phi_sum);




	}
	// 这里没有初始化 corpus_phi_sum，


}

void free_llna_var_param(llna_var_param * v)
{
    gsl_vector_free(v->Ulambda);


    gsl_vector_free(v->Unu);
    gsl_matrix_free(v->phi);
    gsl_matrix_free(v->log_phi);
    gsl_vector_free(v->Utopic_scores);
    free(v);
}
/*
void free_llna_Vvar_param(llna_var_param * v)
{
    gsl_vector_free(v->Vlambda);
    gsl_vector_free(v->Vnu);
    gsl_matrix_free(v->Vphi);
    gsl_matrix_free(v->Vlog_phi);
    gsl_vector_free(v->Vtopic_scores);
    free(v);
}
*/
// 这里是针对一个user文档，一个item文档的
double var_inference(llna_corpus_var * c_var, llna_var_param* var, corpus* all_corpus,
                     llna_model * mod, int d)
{
	gsl_vector * df = gsl_vector_alloc(mod->k);
    double lhood_old = 0;
    double convergence;
    double learn_rate = 0.05;
    doc doc = all_corpus->docs[d];



    //lhood_bnd(c_var, var, mod, all_corpus);  //这里输出mod->lhood
    // 每次循环，采样的triple不同，因此lhood未必会绝对单调递增，大体递增就行
    // 为什么要多次循环，因为多个变量具有相互依赖关系，迭代更新助于收敛。

    do
    {
    	//double aa = exp(1000);
    	//check_nan(aa, "hallo kitty");

    	// 1. sample u i j
    	SampleTriple(all_corpus, var); // get * j
    	show_sample(var->j, var->num_triples);

    	// 2.1 update_u
    	opt_phi(c_var, var, &doc, mod); // 计算 var->phi  顺便更新c_var->corpus_phi_sum
    	df_Ulambda(c_var, var, mod, df, all_corpus);  // df 是导数
    	int is_legal = 1;
    	for (int i = 0; i < mod->k; i++)
    	{
    		check_nan(vget(df, i), "warning: dUlambda is nan");
    		if(isinf(vget(df, i)))
    		{
    			is_legal = 0;
    			break;
    		}
    	}

    	if (is_legal == 1)
    	{
        	gsl_vector_scale(df, learn_rate/var->niter); // df = learn_rate * df
        	gsl_vector_add(var->Ulambda, df);   // lambda = lambda + learn_rate * df
    		show_vect(var->Ulambda, "Ulambda=");
    		gsl_matrix_set_row(c_var->Ucorpus_lambda, var->u, var->Ulambda);
    	}


# if defined(UPDATE_NU)
    	// update_Unu
    	opt_Unu(c_var, var, mod, all_corpus);
    	for (int i = 0; i < mod->k; i++)
    		check_nan(vget(var->Unu, i), "warning:Unu is nan");
    	if (check_nan(vget(var->Unu, 1), "warning:Unu is nan") == 0)
    	{
    		show_vect(var->Unu, "Unu=");
    		gsl_matrix_set_row(c_var->Ucorpus_nu, var->u, var->Unu);
    	}
#endif

    	// 2.2 update_i
    	opt_phi(c_var, var, &doc, mod);
    	df_Ilambda(c_var, var, mod, df, all_corpus);  // df 是导数
    	is_legal = 1;
    	for (int i = 0; i < mod->k; i++)
    	{
    		check_nan(vget(df, i), "warning: dIlambda is nan");
    		if(isinf(vget(df, i)))
    		{
    			is_legal = 0;
    			break;
    		}
    	}
    	if (is_legal == 1)
    	{
        	gsl_vector_scale(df, learn_rate/var->niter); // df = learn_rate * df
        	gsl_vector_add(var->Ilambda, df);   // lambda = lambda + learn_rate * df
    		show_vect(var->Ilambda, "Ilambda=");
    		gsl_matrix_set_row(c_var->Vcorpus_lambda, var->i, var->Ilambda);
    	}


# if defined(UPDATE_NU)
    	// update_Inu
    	opt_Inu(c_var, var, mod, all_corpus);
    	for (int i = 0; i < mod->k; i++)
    		check_nan(vget(var->Inu, 1), "warning:Inu is nan");
    	if (check_nan(vget(var->Inu, 1), "warning:Inu is nan") == 0)
    	{
    		show_vect(var->Inu, "Inu=");
    		gsl_matrix_set_row(c_var->Vcorpus_nu, var->i, var->Inu);
    	}
#endif


# if defined(UPDATE_J)
    	// 2.3 update_j
    	opt_phi(c_var, var, &doc, mod);
    	df_Jlambda(c_var, var, mod, df, all_corpus);  // df 是导数
    	gsl_vector_scale(df, learn_rate); // df = learn_rate * df
    	gsl_vector_add(var->Jlambda, df);//
    	gsl_matrix_set_row(c_var->Vcorpus_lambda, var->j, var->Jlambda);



    	df_Jlambda(c_var, var, mod, df, all_corpus);  // df 是导数
    	is_legal = 1;
    	for (int i = 0; i < mod->k; i++)
    	{
    		check_nan(vget(df, i), "warning: dIlambda is nan");
    		if(isinf(vget(df, i)))
    		{
    			is_legal = 0;
    			break;
    		}

    	}
    	if (is_legal == 1)
    	{
        	gsl_vector_scale(df, learn_rate/var->niter); // df = learn_rate * df
        	gsl_vector_add(Jlambda, df);   // lambda = lambda + learn_rate * df
    		show_vect(var->Ilambda, "Ilambda=");
    		gsl_matrix_set_row(c_var->Vcorpus_lambda, var->i, var->Ilambda);
    	}






# endif
    	lhood_old = var->lhood;
    	printf("lhood:%lf\n", var->lhood);
    	lhood_bnd(c_var, var, mod, all_corpus);

    	var->niter++;

    	//lhood_bnd(var, Udoc, Vdoc,rating, mod);  //重新计算 mod->lhood，也就是下界。当下界稳定时，就停止迭代

    	convergence = fabs((lhood_old - var->lhood) / lhood_old);
    	// printf("lhood = %8.6f (%7.6f)\n", var->lhood, convergence);

    	if ((lhood_old > var->lhood) && (var->niter > 1))
    		printf("WARNING: iter %05d %5.5f > %5.5f\n",
    			   var->niter, lhood_old, var->lhood);
    }
    while (var->niter < PARAMS.var_max_iter);
/*    if (convergence > PARAMS.var_convergence) var->converged = 0;
    else var->converged = 1;*/

/*
#if defined(SHOW_PREDICTION)
    int i,item; double rating,yy;
    gsl_vector Vlambda, Vnu;
    for (i = 0; i<r_ui->nratings; i++ )
    {
    	item = r_ui->items[i];
		rating = r_ui->ratings[i];
		Vlambda = gsl_matrix_row(c_var->Vcorpus_lambda, item-1).vector;
		gsl_blas_ddot(var->Ulambda,&Vlambda,&yy);
		printf("prediction= :%lf;\t true value=%lf\n",yy,rating);

    }
#endif
*/
    gsl_vector_free(df);

    return(var->lhood);
}


void update_expected_ss(llna_corpus_var * c_var, llna_var_param* var, doc* doc, llna_ss* ss)
{
    int i, j, w, c;
    double lilj;

    // 1. covariance and mean suff stats
    for (i = 0; i < ss->Ucov_ss->size1; i++)  // = mod->k
    {
        vinc(ss->Umu_ss, i, vget(var->Ulambda, i));  //gsl_vector_get
        vinc(ss->Vmu_ss, i, vget(var->Ilambda, i));
        for (j = 0; j < ss->Ucov_ss->size2; j++)
        {
            lilj = vget(var->Ulambda, i) * vget(var->Ulambda, j);
            if (i==j)
                mset(ss->Ucov_ss, i, j,
                     mget(ss->Ucov_ss, i, j) + vget(var->Unu, i) + lilj);
            else
                mset(ss->Ucov_ss, i, j, mget(ss->Ucov_ss, i, j) + lilj);

            lilj = vget(var->Ilambda, i) * vget(var->Ilambda, j);
            if (i==j)
                mset(ss->Vcov_ss, i, j,
                     mget(ss->Vcov_ss, i, j) + vget(var->Inu, i) + lilj);
            else
                mset(ss->Vcov_ss, i, j, mget(ss->Vcov_ss, i, j) + lilj);

        }
    }


    // 2. topics suff stats
    for (i = 0; i < doc->nterms; i++)
    {
        for (j = 0; j < ss->beta_ss->size1; j++)
        {
            w = doc->word[i];
            c = doc->count[i];
            mset(ss->beta_ss, j, w,
                 mget(ss->beta_ss, j, w) + c * mget(var->phi, i, j));
        }
    }
    // 4. number of data
    ss->ndata++;  //最终应该等于 ndocs
}

/*
void update_Vexpected_ss(llna_var_param* var, doc* Vdoc, ratings * r_vj,llna_corpus_var * c_var, llna_ss* ss)
{
    int i, j, w, c;
    double lilj;

    // 1.covariance and mean suff stats
    for (i = 0; i < ss->Vcov_ss->size1; i++)
    {
        vinc(ss->Vmu_ss, i, vget(var->Vlambda, i));  //gsl_vector_get
        for (j = 0; j < ss->Vcov_ss->size2; j++)
        {
            lilj = vget(var->Vlambda, i) * vget(var->Vlambda, j);
            if (i==j)
                mset(ss->Vcov_ss, i, j,
                     mget(ss->Vcov_ss, i, j) + vget(var->Vnu, i) + lilj);
            else
                mset(ss->Vcov_ss, i, j, mget(ss->Vcov_ss, i, j) + lilj);
        }
    }
    // 2.topics suff stats
    for (i = 0; i < Vdoc->nterms; i++)
    {
        for (j = 0; j < ss->Vbeta_ss->size1; j++)
        {
            w = Vdoc->word[i];
            c = Vdoc->count[i];
            mset(ss->Vbeta_ss, j, w,
                 mget(ss->Vbeta_ss, j, w) + c * mget(var->Vphi, i, j));
        }
    }

    // 3. ratings covariance suff stats
    int user;
    double rating,uv,t1,t2,t3;
    gsl_vector Ulambda, Unu;

    gsl_blas_dcopy(var->Vlambda,temp[0]);  //temp[0]=Vlambda^2
    for (i = 0; i < var->Vlambda->size ; i++) {
    	t1 = gsl_vector_get(temp[0],i)*gsl_vector_get(temp[0],i);
    	gsl_vector_set(temp[0],i,t1);
    }
    for(i = 0; i < r_vj->nratings; i++)
    {
    	user = r_vj->users[i];
    	rating = r_vj->ratings[i];
        Ulambda = gsl_matrix_row(c_var->Ucorpus_lambda, user-1).vector;
        Unu = gsl_matrix_row(c_var->Ucorpus_nu, user-1).vector;

        gsl_blas_ddot(&Ulambda,var->Vlambda,&uv);
        gsl_blas_dcopy(&Ulambda,temp[1]);  //temp[1]=Ulambda^2
        for (j = 0; j < Ulambda.size ; j++) {
        	t1 = gsl_vector_get(temp[1],j)*gsl_vector_get(temp[1],j);
        	gsl_vector_set(temp[1],j,t1);
        }
        gsl_blas_ddot(temp[0],&Unu, &t1);  //t1=Ulambda^2 *Vnu
        gsl_blas_ddot(temp[1],var->Vnu, &t2);  //t2=Vlambda^2 *Unu
        gsl_blas_ddot(&Unu,var->Vnu, &t3);  //t3=Unu * Vnu

        ss->cov_ss += rating*rating + uv*uv + t1 + t2 + t3 - 2*rating*uv;
        ss->nratings++;

    }

    // 4. number of data
    ss->Vndata++;  //最终应该等于 nratings
}
*/

/*
 * importance sampling the likelihood based on the variational posterior
 *
 */

double sample_term(llna_var_param* var, doc* d, llna_model* mod, double* eta)
{/*
    int i, j, n;
    double t1, t2, sum, theta[mod->k];
    double word_term;

    t1 = (0.5) * mod->log_det_inv_cov;
    t1 += - (0.5) * (mod->k) * 1.837877;
    for (i = 0; i < mod->k; i++)
        for (j = 0; j < mod->k ; j++)
            t1 -= (0.5) *
                (eta[i] - vget(mod->mu, i)) *
                mget(mod->inv_cov, i, j) *
                (eta[j] - vget(mod->mu, j));

    // compute theta
    sum = 0;
    for (i = 0; i < mod->k; i++)
    {
        theta[i] = exp(eta[i]);
        sum += theta[i];
    }
    for (i = 0; i < mod->k; i++)
    {
        theta[i] = theta[i] / sum;
    }

    // compute word probabilities
    for (n = 0; n < d->nterms; n++)
    {
        word_term = 0;
        for (i = 0; i < mod->k; i++)
            word_term += theta[i]*exp(mget(mod->log_beta,i,d->word[n]));
        t1 += d->count[n] * safe_log(word_term);
    }

    // log(q(\eta | lambda, nu))
    t2 = 0;
    for (i = 0; i < mod->k; i++)
        t2 += log(gsl_ran_gaussian_pdf(eta[i] - vget(var->lambda,i), sqrt(vget(var->nu,i))));
    return(t1-t2);*/
	return(0);
}


double sample_lhood(llna_var_param* var, doc* d, llna_model* mod)
{/*
    int nsamples, i, n;
    double eta[mod->k];
    double log_prob, sum = 0, v;
    gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);

    gsl_rng_set(r, (long) 1115574245);
    nsamples = 10000;

    // for each sample
    for (n = 0; n < nsamples; n++)
    {
        // sample eta from q(\eta)
        for (i = 0; i < mod->k; i++)
        {
            v = gsl_ran_gaussian_ratio_method(r, sqrt(vget(var->nu,i)));
            eta[i] = v + vget(var->lambda, i);
        }
        // compute p(w | \eta) - q(\eta)
        log_prob = sample_term(var, d, mod, eta);
        // update log sum
        if (n == 0) sum = log_prob;
        else sum = log_sum(sum, log_prob);
        // printf("%5.5f\n", (sum - log(n+1)));
    }
    sum = sum - log((double) nsamples);
    return(sum);*/
	return(0);
}


/*
 * expected theta under a variational distribution
 *
 * (v is assumed allocated to the right length.)
 *
 */


void expected_theta(llna_var_param *var, doc* d, llna_model *mod, gsl_vector* val)
{/*
    int nsamples, i, n;
    double eta[mod->k];
    double theta[mod->k];
    double e_theta[mod->k];
    double sum, w, v;
    gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);

    gsl_rng_set(r, (long) 1115574245);
    nsamples = 100;

    // initialize e_theta
    for (i = 0; i < mod->k; i++) e_theta[i] = -1;
    // for each sample
    for (n = 0; n < nsamples; n++)
    {
        // sample eta from q(\eta)
        for (i = 0; i < mod->k; i++)
        {
            v = gsl_ran_gaussian_ratio_method(r, sqrt(vget(var->nu,i)));
            eta[i] = v + vget(var->lambda, i);
        }
        // compute p(w | \eta) - q(\eta)
        w = sample_term(var, d, mod, eta);
        // compute theta
        sum = 0;
        for (i = 0; i < mod->k; i++)
        {
            theta[i] = exp(eta[i]);
            sum += theta[i];
        }
        for (i = 0; i < mod->k; i++)
            theta[i] = theta[i] / sum;
        // update e_theta
        for (i = 0; i < mod->k; i++)
            e_theta[i] = log_sum(e_theta[i], w +  safe_log(theta[i]));
    }
    // normalize e_theta and set return vector
    sum = -1;
    for (i = 0; i < mod->k; i++)
    {
        e_theta[i] = e_theta[i] - log(nsamples);
        sum = log_sum(sum, e_theta[i]);
    }
    for (i = 0; i < mod->k; i++)
        vset(val, i, exp(e_theta[i] - sum));*/
}

/*
 * log probability of the document under proportions theta and topics
 * beta
 *
 */

double log_mult_prob(doc* d, gsl_vector* theta, gsl_matrix* log_beta)
{
    int i, k;
    double ret = 0;
    double term_prob;

    for (i = 0; i < d->nterms; i++)
    {
        term_prob = 0;
        for (k = 0; k < log_beta->size1; k++)
            term_prob += vget(theta, k) * exp(mget(log_beta, k, d->word[i]));
        ret = ret + safe_log(term_prob) * d->count[i];
    }
    return(ret);
}


/*
 * writes the word assignments line for a document to a file
 *
 */

void write_word_assignment(FILE* f, doc* d, gsl_matrix* phi)
{
    int n;

    fprintf(f, "%03d", d->nterms);
    for (n = 0; n < d->nterms; n++)
    {
        gsl_vector phi_row = gsl_matrix_row(phi, n).vector;
        fprintf(f, " %04d:%02d", d->word[n], argmax(&phi_row));
    }
    fprintf(f, "\n");
    fflush(f);
}

/*
 * write corpus variational parameter
 *
 */

void write_c_var(llna_corpus_var * c_var, char * root)  //已经改完
{
    char filename[200];

    sprintf(filename, "%s-Ucorpus_lambda.dat", root);
    printf_matrix(filename, c_var->Ucorpus_lambda);

    sprintf(filename, "%s-Vcorpus_lambda.dat", root);
    printf_matrix(filename, c_var->Vcorpus_lambda);

    sprintf(filename, "%s-Ucorpus_nu.dat", root);
    printf_matrix(filename, c_var->Ucorpus_nu);

    sprintf(filename, "%s-Vcorpus_nu.dat", root);
    printf_matrix(filename, c_var->Vcorpus_nu);

    sprintf(filename, "%s-corpus_phi_sum.dat", root);
    printf_matrix(filename, c_var->corpus_phi_sum);

}
