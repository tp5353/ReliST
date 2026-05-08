args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 13 || length(args) > 15) {
  stop(
    paste(
      "Expected 13 to 15 args:",
      "spatial_mtx spatial_genes spatial_barcodes spatial_coords",
      "reference_mtx reference_genes reference_cells reference_celltypes",
      "weights_csv uncertainty_csv results_df_csv",
      "max_cores doublet_mode [umi_min] [cell_min_instance]"
    )
  )
}

suppressPackageStartupMessages({
  library(Matrix)
  library(spacexr)
})

spatial_mtx_path <- args[[1]]
spatial_genes_path <- args[[2]]
spatial_barcodes_path <- args[[3]]
spatial_coords_path <- args[[4]]
reference_mtx_path <- args[[5]]
reference_genes_path <- args[[6]]
reference_cells_path <- args[[7]]
reference_celltypes_path <- args[[8]]
weights_csv_path <- args[[9]]
uncertainty_csv_path <- args[[10]]
results_df_csv_path <- args[[11]]
max_cores <- as.integer(args[[12]])
doublet_mode <- args[[13]]
umi_min <- if (length(args) >= 14) as.numeric(args[[14]]) else 100
cell_min_instance <- if (length(args) >= 15) as.integer(args[[15]]) else 25

spatial_counts <- readMM(spatial_mtx_path)
spatial_counts <- as(spatial_counts, "dgCMatrix")
rownames(spatial_counts) <- readLines(spatial_genes_path)
colnames(spatial_counts) <- readLines(spatial_barcodes_path)

reference_counts <- readMM(reference_mtx_path)
reference_counts <- as(reference_counts, "dgCMatrix")
rownames(reference_counts) <- readLines(reference_genes_path)
colnames(reference_counts) <- readLines(reference_cells_path)

coords <- read.csv(spatial_coords_path, row.names = 1, check.names = FALSE)
coords <- coords[colnames(spatial_counts), , drop = FALSE]

celltypes_df <- read.csv(reference_celltypes_path, row.names = 1, check.names = FALSE)
celltypes_df <- celltypes_df[colnames(reference_counts), , drop = FALSE]
cell_types <- factor(celltypes_df$cell_type)
names(cell_types) <- colnames(reference_counts)

spatial_rna <- SpatialRNA(
  coords = coords,
  counts = spatial_counts,
  nUMI = Matrix::colSums(spatial_counts),
  require_int = TRUE
)
reference <- Reference(
  counts = reference_counts,
  cell_types = cell_types,
  nUMI = Matrix::colSums(reference_counts),
  require_int = TRUE,
  min_UMI = umi_min
)

rctd <- create.RCTD(
  spatialRNA = spatial_rna,
  reference = reference,
  max_cores = max_cores,
  UMI_min = umi_min,
  CELL_MIN_INSTANCE = cell_min_instance
)
rctd <- run.RCTD(rctd, doublet_mode = doublet_mode)

weights <- rctd@results$weights
weights <- normalize_weights(weights)
weights_df <- as.data.frame(as.matrix(weights))
weights_df$spot_id <- rownames(weights_df)
weights_df <- weights_df[, c("spot_id", setdiff(colnames(weights_df), "spot_id"))]
write.csv(weights_df, weights_csv_path, row.names = FALSE, quote = FALSE)

uncertainty <- 1 - apply(as.matrix(weights), 1, max)
uncertainty_df <- data.frame(
  spot_id = names(uncertainty),
  rctd_inverse_max_weight = uncertainty,
  row.names = NULL,
  check.names = FALSE
)
write.csv(uncertainty_df, uncertainty_csv_path, row.names = FALSE, quote = FALSE)

results_df <- rctd@results$results_df
results_df$spot_id <- rownames(results_df)
results_df <- results_df[, c("spot_id", setdiff(colnames(results_df), "spot_id"))]
write.csv(results_df, results_df_csv_path, row.names = FALSE, quote = FALSE)
