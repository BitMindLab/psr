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

/*************************************************************************
 *
 * llna.c
 *
 * estimation of an llna model by variational em
 *
 *************************************************************************/

#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <gsl/gsl_blas.h>
#include <assert.h>

#include "corpus.h"
#include "ctm.h"
#include "inference.h"
#include "gsl-wrappers.h"
#include "params.h"
#include "omp.h"

extern llna_params PARAMS;

/*
 * e step
 *
 */


/*void expectation(corpus* corpus, llna_model* model, llna_ss* ss,
                 double* avg_niter, double* total_lhood,
                 gsl_matrix* corpus_lambda, gsl_matrix* corpus_nu,
                 gsl_matrix* corpus_phi_sum,
                 short reset_var, double* converged_pct)*/

void expectation(corpus* all_corpus, llna_model* model, llna_ss* ss,
                 double* avg_niter, double* total_lhood,
                 llna_corpus_var * c_var, double* converged_pct)
{
    double lhood, total;

    *avg_niter = 0.0;
    *converged_pct = 0;

    for(int i = 0; i < 1; i++)
    {
    	total=0;
    	reset_llna_ss(ss);


# if defined(PARALLEL)
#pragma omp parallel for num_threads(16)
# endif
    	for (int d = 0; d < all_corpus->ndocs; d++)
    	{
    		new_temp_vectors(model->k);
    		llna_var_param* var;
			printf("doc ID: %5d, ThreadId = %d\n ", d+1, omp_get_thread_num());

			doc doc = all_corpus->docs[d];  //all_corpus->docs 是指针，all_corpus->docs[d]已经不是指针啦
    		var = new_llna_var_param(doc.nterms, model->k);
    		init_var(c_var, var, &doc, model);

    		// lhood
			lhood = var_inference(c_var, var, all_corpus, model, d);  //---------很重要：更新变分参数, doc+model=>var-------
		    update_expected_ss(c_var, var, &doc, ss);   //更新的ss,用于M步的参数估计
			total += lhood;
			printf("lhood %5.5e   niter %5d\n", lhood, var->niter);
			*avg_niter += var->niter;           //avg_niter=var_inference过程中的平均迭代次数
			*converged_pct += var->converged;

			free_llna_var_param(var);
			free_temp_vectors(model->k);
    	}
    }
    //gsl_vector_free();
    //gsl_vector_free(Vphi_sum);
    *avg_niter = *avg_niter / all_corpus->ndocs; //这里是否修改，没太大意义
    *converged_pct = *converged_pct / all_corpus->ndocs; //这里意义也不大
    *total_lhood = total;
}


/*
 * m step
 *
 */

void cov_shrinkage(gsl_matrix* mle, int n, gsl_matrix* result)
{
    int p = mle->size1, i;
    double alpha = 0, tau = 0, log_lambda_s = 0;
    gsl_vector
        *lambda_star = gsl_vector_calloc(p),
        t, u,
        *eigen_vals = gsl_vector_calloc(p),
        *s_eigen_vals = gsl_vector_calloc(p);
    gsl_matrix
        *d = gsl_matrix_calloc(p,p),
        *eigen_vects = gsl_matrix_calloc(p,p),
        *s_eigen_vects = gsl_matrix_calloc(p,p),
        *result1 = gsl_matrix_calloc(p,p);

    // get eigen decomposition

    sym_eigen(mle, eigen_vals, eigen_vects);
    for (i = 0; i < p; i++)
    {

        // compute shrunken eigenvalues

        alpha = 1.0/(n+p+1-2*i);
        vset(lambda_star, i, n * alpha * vget(eigen_vals, i));
    }

    // get diagonal mle and eigen decomposition

    t = gsl_matrix_diagonal(d).vector;
    u = gsl_matrix_diagonal(mle).vector;
    gsl_vector_memcpy(&t, &u);
    sym_eigen(d, s_eigen_vals, s_eigen_vects);

    // compute tau^2

    for (i = 0; i < p; i++)
        log_lambda_s += log(vget(s_eigen_vals, i));
    log_lambda_s = log_lambda_s/p;
    for (i = 0; i < p; i++)
        tau += pow(log(vget(lambda_star, i)) - log_lambda_s, 2)/(p + 4) - 2.0 / n;

    // shrink \lambda* towards the structured eigenvalues

    for (i = 0; i < p; i++)
        vset(lambda_star, i,
             exp((2.0/n)/((2.0/n) + tau) * log_lambda_s +
                 tau/((2.0/n) + tau) * log(vget(lambda_star, i))));

    // put the eigenvalues in a diagonal matrix

    t = gsl_matrix_diagonal(d).vector;
    gsl_vector_memcpy(&t, lambda_star);

    // reconstruct the covariance matrix

    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, d, eigen_vects, 0, result1);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, eigen_vects, result1, 0, result);

    // clean up

    gsl_vector_free(lambda_star);
    gsl_vector_free(eigen_vals);
    gsl_vector_free(s_eigen_vals);
    gsl_matrix_free(d);
    gsl_matrix_free(eigen_vects);
    gsl_matrix_free(s_eigen_vects);
    gsl_matrix_free(result1);
}



void maximization(llna_model* model, llna_ss * ss)
{
    int i, j;
    double sum;

# if defined(UPDATE_MOD)
    // 1. mean maximization  更新 model->mu
    for (i = 0; i < model->k; i++) {
    	vset(model->Umu, i, vget(ss->Umu_ss, i) / ss->nrating);
    	vset(model->Vmu, i, vget(ss->Vmu_ss, i) / ss->nrating);
    }

    // 2.1 covariance maximization
    for (i = 0; i < model->k; i++)
    {
        for (j = 0; j < model->k; j++)
        {
            mset(model->Ucov, i, j,
                 (1.0 / ss->nrating) *
                 (mget(ss->Ucov_ss, i, j) +
                  ss->nrating * vget(model->Umu, i) * vget(model->Umu, j) -
                  vget(ss->Umu_ss, i) * vget(model->Umu, j) -
                  vget(ss->Umu_ss, j) * vget(model->Umu, i)));
        }
    }
    if (PARAMS.cov_estimate == SHRINK)
    {
        cov_shrinkage(model->Ucov, ss->nrating, model->Ucov);
    }
    matrix_inverse(model->Ucov, model->Uinv_cov);
    model->Ulog_det_inv_cov = log_det(model->Uinv_cov);

    // 2.2 covariance maximization
    for (i = 0; i < model->k; i++)
    {
        for (j = 0; j < model->k; j++)
        {
            mset(model->Vcov, i, j,
                 (1.0 / ss->nrating) *
                 (mget(ss->Vcov_ss, i, j) +
                  ss->nrating * vget(model->Vmu, i) * vget(model->Vmu, j) -
                  vget(ss->Vmu_ss, i) * vget(model->Vmu, j) -
                  vget(ss->Vmu_ss, j) * vget(model->Vmu, i)));
        }
    }
    if (PARAMS.cov_estimate == SHRINK)
    {
        cov_shrinkage(model->Vcov, ss->nrating, model->Vcov);
    }
    matrix_inverse(model->Vcov, model->Vinv_cov);
    model->Vlog_det_inv_cov = log_det(model->Vinv_cov);


# endif

    // 3. topic maximization  更新 model->log_beta
    for (i = 0; i < model->k; i++)
    {
        sum = 0;
        for (j = 0; j < model->log_beta->size2; j++)
            sum += mget(ss->beta_ss, i, j);

        if (sum == 0) sum = safe_log(sum) * model->log_beta->size2;
        else sum = safe_log(sum);

        for (j = 0; j < model->log_beta->size2; j++)
            mset(model->log_beta, i, j, safe_log(mget(ss->beta_ss, i, j)) - sum);
    }
}


/*
 * run em
 *
 */
// 根据start的值<rand/seed/model> ，来初始化模型
llna_model* em_initial_model(int k, corpus* all_corpus, char* start)
{
    llna_model* model;
    printf("starting from %s\n", start);
    if (strcmp(start, "rand")==0)
        model = random_init(k, all_corpus->nterms, all_corpus->nuser, all_corpus->nitem);  //这是随机初始化，0均值，单位阵方差，beta为随机值
/*
    else if (strcmp(start, "seed")==0)
        model = corpus_init(k, corpus);    //这是固定种子随机初始化
*/
    else
        model = read_llna_model(start,all_corpus->nuser, all_corpus->nitem);
    return(model);
}


void em(char* dataset, int k, char* start, char* dir)
{
    FILE* lhood_fptr;
    char string[100];
    int iteration = 0;
    double convergence = 1, lhood = 0, lhood_old = 0;
    corpus* all_corpus;
    llna_model *model;
    llna_ss* ss;  //这应该也需要更改
    time_t t1,t2;
    double avg_niter, converged_pct, old_conv = 0;

    llna_corpus_var * c_var;

    // read the data and make the directory

    all_corpus = read_data(dataset);  //数据格式类似lda-c
    //corpus = read_data(dataset_item);  //--
    //r = read_rating(dataset_rating); // 数据格式R×3

    mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

    // set up the log likelihood log file

    sprintf(string, "%s/likelihood.dat", dir);
    lhood_fptr = fopen(string, "w");

    // run em
    model = em_initial_model(k, all_corpus, start); //因为只采用random初始化，因此只用到了topic数目k
    ss = new_llna_ss(model);    //申请ss内部变量存储空间空间,也需要改---还未改----------------------

    //corpus_lambda = gsl_matrix_alloc(corpus->ndocs, model->k); //--
    c_var = new_llna_corpus_var(all_corpus->nuser, all_corpus->nitem, all_corpus->ndocs, model->k);
    init_corpus_var(c_var, start);

    time(&t1);

    int iter_start;
    if (atoi(start) != NULL) {
    	iter_start = atoi(start);
    } else {
    	iter_start = 0;
    }

    // write start model & c_var & rmse_start
    sprintf(string, "%s/start-%03d", dir, iteration+iter_start);  //这是确认
    write_llna_model(model, string);  //已改
    write_c_var(c_var, string);

    //===========这里是核心框架=======
    do
    {

        printf("***** EM ITERATION %d *****\n", iteration);

        //===========E-step=======
        expectation(all_corpus, model, ss, &avg_niter, &lhood,
                    c_var, &converged_pct);
        time(&t2);
        convergence = (lhood_old - lhood) / lhood_old; //lhood 应该要大于 lhood_old，才趋向收敛
        fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld %5.5f %1.5f\n",
                iteration, lhood, convergence, (int) t2 - t1, avg_niter, converged_pct);

        printf("convergence is :%f\n",convergence);

		//===========M-step=======
		maximization(model, ss);

		lhood_old = lhood;
		iteration++;


		if (((iteration % PARAMS.lag)==0) || isnan(lhood))
		{
			sprintf(string, "%s/%03d", dir, iteration + iter_start);
			write_llna_model(model, string);
			write_c_var(c_var, string);
		}
		time(&t1);

        //}

        fflush(lhood_fptr);


        old_conv = convergence;
    }
    while ((iteration < PARAMS.em_max_iter));
/*    while ((iteration < PARAMS.em_max_iter) &&
    		((convergence > 0.003) || (convergence < 0)));*/
           //((convergence > PARAMS.em_convergence) || (convergence < 0)));

    sprintf(string, "%s/final", dir);
    write_llna_model(model, string);
    write_c_var(c_var, string);

    sprintf(string, "%s/final-iteration.dat", dir);
    printf_value(string,iteration-1);


    fclose(lhood_fptr);
}


/*
 * load a model, and do approximate inference for each document in a corpus
 * inference过程中，corpus_user和item都不变，只不过是给定了一些新的user-item对，来预测其评分而已
 * pre@N   recall@N F1@N
 */

void inference(char* all_dataset, char* test_dataset, int N, char* model_root, char* out)
{
    int nusers, nitems, iteration;
    double precision, recall;
    char fname[100];
    corpus* all_corpus, * test_corpus;

    // read the data and model
    all_corpus = read_data(all_dataset);
    test_corpus = read_data(test_dataset);
    nusers = all_corpus->nuser;
	nitems = all_corpus->nitem;
    gsl_matrix * predict_r = gsl_matrix_calloc(nusers, nitems);
    //gsl_matrix * rec_list = gsl_matrix_calloc(N, 2); //第一列是user，第二列是item
    vect rect_u, rect_i;
    rect_u.size = N;
    rect_i.size = N;
    rect_u.id = malloc(sizeof(int) * N);
    rect_i.id = malloc(sizeof(int) * N);


    //predict_r = gsl_vector_alloc(ratings->nratings);

    llna_model * model = read_llna_model("./model/001",nusers,nitems);

    //sprintf(fname, "%s-iteration.dat", model_root);
    //scanf_value(fname, &iteration);

    gsl_matrix * Ucorpus_lambda = gsl_matrix_alloc(nusers, model->k);
    gsl_matrix * Vcorpus_lambda = gsl_matrix_alloc(nitems, model->k);



    //for(int i=0;i<=iteration+1;i++)
    for(int i = 1; i <= 100; i++)
    {
        sprintf(fname, "./model/%03d-Ucorpus_lambda.dat", i);
        scanf_matrix(fname, Ucorpus_lambda);

        sprintf(fname, "./model/%03d-Vcorpus_lambda.dat", i);
        scanf_matrix(fname, Vcorpus_lambda);

        // get precision recall @ N
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Ucorpus_lambda, Vcorpus_lambda, 0.0, predict_r);

        for (int j = 0; j < N; j++ )
        {
        	size_t u_id, i_id;
        	gsl_matrix_max_index(predict_r, &u_id, &i_id);
        	mset(predict_r, u_id, i_id, -1000.0);
        	rect_u.id[j] = u_id;  // 这里不需要-1，本身就是从0开始的
        	rect_i.id[j] = i_id;
        }

        evaluate(test_corpus, rect_u, rect_i, &precision, &recall, N);
        double F1 = 2 * precision * recall / (precision + recall);
        printf("iteration:%03d,inference:\tprecision=%lf\trecall=%lf\tF1=%lf\n",i, precision, recall, F1);

    }

    gsl_matrix_free(Ucorpus_lambda);
    gsl_matrix_free(Vcorpus_lambda);


}


/*
 * split documents into two random parts
 *
 */

void within_doc_split(char* dataset, char* src_data, char* dest_data, double prop)
{
    int i;
    corpus * corp, * dest_corp;

    corp = read_data(dataset);
    dest_corp = malloc(sizeof(corpus));
    printf("splitting %d docs\n", corp->ndocs);
    dest_corp->docs = malloc(sizeof(doc) * corp->ndocs);
    dest_corp->nterms = corp->nterms;
    dest_corp->ndocs = corp->ndocs;
    for (i = 0; i < corp->ndocs; i++)
        split(&(corp->docs[i]), &(dest_corp->docs[i]), prop);
    write_corpus(dest_corp, dest_data);
    write_corpus(corp, src_data);
}


/*
 * for each partially observed document: (a) perform inference on the
 * observations (b) take expected theta and compute likelihood
 *
 */

int pod_experiment(char* observed_data, char* heldout_data,
                   char* model_root, char* out)
{/*
    corpus *obs, *heldout;
    llna_model *model;
    llna_var_param *var;
    int i;
    gsl_vector *log_lhood, *e_theta;
    doc obs_doc, heldout_doc;
    char string[100];
    double total_lhood = 0, total_words = 0, l;
    FILE* e_theta_file = fopen("/Users/blei/llna050_e_theta.txt", "w");

    // load model and data
    obs = read_data(observed_data);
    heldout = read_data(heldout_data);
    assert(obs->ndocs == heldout->ndocs);
    model = read_llna_model(model_root);

    // run experiment
    init_temp_vectors(model->k-1); // !!! hacky
    log_lhood = gsl_vector_alloc(obs->ndocs + 1);
    e_theta = gsl_vector_alloc(model->k);
    for (i = 0; i < obs->ndocs; i++)
    {
        // get observed and heldout documents
        obs_doc = obs->docs[i];
        heldout_doc = heldout->docs[i];
        // compute variational distribution
        var = new_llna_var_param(obs_doc.nterms, model->k);
        init_var_unif(var, &obs_doc, model);
        var_inference(var, &obs_doc, model);
        expected_theta(var, &obs_doc, model, e_theta);

        vfprint(e_theta, e_theta_file);

        // approximate inference of held out data
        l = log_mult_prob(&heldout_doc, e_theta, model->log_beta);
        vset(log_lhood, i, l);
        total_words += heldout_doc.total;
        total_lhood += l;
        printf("hid doc %d    log_lhood %5.5f\n", i, vget(log_lhood, i));
        // save results?
        free_llna_var_param(var);
    }
    vset(log_lhood, obs->ndocs, exp(-total_lhood/total_words));
    printf("perplexity : %5.10f", exp(-total_lhood/total_words));
    sprintf(string, "%s-pod-llna.dat", out);
    printf_vector(string, log_lhood);*/
    return(0);
}


/*
 * little function to count the words in each document and spit it out
 *
 */

void count(char* corpus_name, char* output_name)
{
    corpus *c;
    int i;
    FILE *f;
    int j;
    f = fopen(output_name, "w");
    c = read_data(corpus_name);
    for (i = 0; i < c->ndocs; i++)
    {
        j = c->docs[i].total;
        fprintf(f, "%5d\n", j);
    }
}

/*
 * main function
 *
 */

int main(int argc, char* argv[])
{
    if (argc > 1)
    {
        if (strcmp(argv[1], "est")==0)
        {
            read_params(argv[6]);
            print_params();

            em(argv[2], atoi(argv[3]), argv[4], argv[5]);

            //inference(argv[2], argv[2], 2000, "final","out"); //ap_User_URL.dat
            inference(argv[2], argv[2], 20000, "final","out"); //ap_User_URL_mid_train.dat

            return(0);
        }
        if (strcmp(argv[1], "inf")==0)
        {
            read_params(argv[7]);
            print_params();
            //inference(argv[2], argv[3], argv[4],argv[5],argv[6]);
            return(0);
        }
    }
    printf("usage : ctm est <dataset> <# topics> <rand/seed/model> <dir> <settings>\n");
    //printf("        ctm inf <dataset> <model-prefix> <results-prefix> <settings>\n");
    //error_end:printf("program is ended");
    return(0);
}
