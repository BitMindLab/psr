'''
Created on Aug 18, 2013

@author: xx
'''

import sys, numpy

def print_topics(beta_file, vocab_file,
                 nwords = 25, out = sys.stdout):

    # get the vocabulary

    vocab = file(vocab_file, 'r').readlines()
    #vocab = map(lambda x: x.split()[1], vocab)

    indices = range(len(vocab))
    topic = numpy.array(map(float, file(beta_file, 'r').readlines()))

    nterms  = len(vocab)
    ntopics = len(topic)/nterms
    topic   = numpy.reshape(topic, [ntopics, nterms])
    f = open(r'topic','w')
    for i in range(ntopics):
        out.write('\ntopic %03d\n' % i)
        f.writelines('\ntopic %03d\n' % i)
        indices.sort(lambda x,y: -cmp(topic[i,x], topic[i,y]))
        for j in range(nwords):
            
            out.write('     %4.2f     %s' % (topic[i,indices[j]],  vocab[indices[j]]))
            f.writelines('     '+vocab[indices[j]])
    f.close()



if (__name__ == '__main__'):
     beta_file = sys.argv[1]
     vocab_file = sys.argv[2]
     nwords = int(sys.argv[3])
     print_topics(beta_file, vocab_file, nwords)
