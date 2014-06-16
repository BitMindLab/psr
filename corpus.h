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

#ifndef CORPUS_H
#define CORPUS_H

#include <math.h>

#define OFFSET 0

/*
 * a document is a collection of counts and terms
 *
 */

typedef struct doc {
    int total;
    int nterms;
    int d_id;
    int u_id;
    int v_id;
    int * word;
    int * count;

} doc;

/*
 * vector with integer element
 *
 */

typedef struct vect {
    int * id;
    int size;
} vect;

/*
 * a corpus is a collection of documents
 *
 */

typedef struct corpus {
    doc* docs;
    int nterms;
    int ndocs;
    int nuser;
    int nitem;
    vect * usermatrix; //user_items, 也可以用稀疏矩阵来表示，但是我不熟悉
    vect * itemmatrix;
    vect * udoc_list;
    vect * idoc_list;

} corpus;




/*
 * implicit ratings (rating matrix)
 *
 */

/*typedef struct ratings {
	u_like * u_like;
	//int nuser;  // = corpus->nuser
} ratings;*/



/*
 * functions
 *
 */

corpus* read_data(const char*);
void print_doc(doc* d);
void split(doc* orig, doc* dest, double prop);
void write_corpus(corpus* c, char* filename);


#endif
