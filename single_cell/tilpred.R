library(Matrix)
library(anndata)
library(TILPRED)
library(data.table)
library(SingleCellExperiment)


convert_adata <- function(adata){
  counts <- adata$X
  obs <- adata$obs
  var <- adata$var
  counts <- counts[rownames(obs), ]
  counts <- Matrix::t(counts)
  sce <- SingleCellExperiment(assay=list(counts=counts),
                              rowData=var,
                              colData=obs)
  sce
}

adata <- read_h5ad("sc_dataset.h5ad")
# sce <- convert_adata(adata)
sce <- SingleCellExperiment(assays = list(logcounts=t(as.matrix(adata$X))))
x <- predictTilState(sce)
write.csv(x$predictedState, "tcell_annot.csv")
