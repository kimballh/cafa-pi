# The code from protein_embeddings.Rmd, but in a script.

if (!require("pacman")) install.packages("pacman")
pacman::p_load("tidyverse")

go_vecs <- read_csv("../data/parsed/go_embeddings.csv")
proteins <- read_csv("../data/parsed/training.csv") %>% select(Sequence, GO_ID)

set.seed(0)
seq_embeddings <- proteins %>% 
  left_join(go_vecs) %>% 
  select(-GO_ID) %>% 
  group_by(Sequence) %>% 
  summarize_all(mean, na.rm = TRUE) %>% 
  mutate_all(funs(replace(., is.na(.), rnorm(1, 0, 0.01))))

seq_embeddings %>% write_csv("../data/parsed/seq_embeddings.csv")