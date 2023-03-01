import benchmark_embeddings
import get_word_embeddings

vocab = benchmark_embeddings.get_vocab()

# For best embeddings obtained so far, we used these hyper-parameters:
#SYN_ANT_MULT=3 ,HYPER_HYPO_MULT=2 ,POS_MULT= 1,TEST_EPOCHS=210000
w2i,i2w,i2v, embeds = get_word_embeddings.get_embeds(vocab,3,2,1,210000)
benchmark_embeddings.test_embeddings(w2i,embeds)