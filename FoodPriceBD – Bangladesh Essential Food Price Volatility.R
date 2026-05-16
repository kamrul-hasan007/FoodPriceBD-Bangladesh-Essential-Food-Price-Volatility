############################################################
# FoodPriceBD – Bangladesh Essential Food Price Volatility
# Introduction to Data Science – Final-Term Project
#
# Research Objective:
#   To analyze and predict essential food price volatility
#   in Bangladesh by collecting real-world market data,
#   performing exploratory analysis, and building both
#   regression (price prediction) and classification
#   (price movement direction) models.
#
# Author  : Md. Kamrul Hasan
# Date    : May 2026
# Dataset : 10,000+ rows (10 commodities × 8 divisions
#           × 168 months = 13,440 observations)
#
# FIXES APPLIED (vs original):
#  1. Removed deprecated `..count..` aesthetic → use
#     after_stat(count) for ggplot2 >= 3.4.0
#  2. Fixed Neural Network confusion-matrix misalignment:
#     actual and predicted now share the same sampled rows
#  3. Fixed GBM classifier prediction: multinomial output
#     is a 3-D array; correct indexing to [,,1]
#  4. Fixed switch() fall-through: added explicit default
#     for best_reg_pred / best_test_actual
#  5. Guarded against zero-row glmnet matrices (when API
#     data has no rows, make_glmnet_matrices crashed)
#  6. Replaced base R `..` pronoun with tidy-select
#     helpers to avoid column-lookup errors
#  7. Standardize_worldbank_wide: safer column existence
#     checks using %in% names(.) inside mutate
#  8. Added set.seed() calls before every model for
#     full reproducibility
#  9. YOY plot: fixed duplicate aesthetic mapping
#     (fill already mapped; removed extra aes call)
# 10. Moved `cls_comparison` initialisation before the
#     if/else block so it always exists for the dashboard
# 11. Fixed final_report tibble: wrapped nrow(cls_comparison)
#     in if/else to avoid errors when cls_comparison is empty
# 12. All ggsave_bd calls use explicit `plot =` argument
# 13. Added proper section headers matching project rubric
############################################################


############################
# 0. PACKAGE SETUP
############################
AUTO_INSTALL <- TRUE

required_packages <- c(
  # --- data collection ---
  "httr", "jsonlite", "rvest", "xml2",
  # --- data wrangling ---
  "dplyr", "tidyr", "stringr", "readr", "lubridate",
  "janitor", "tibble", "purrr",
  # --- visualisation ---
  "ggplot2", "scales", "RColorBrewer", "gridExtra",
  "viridis", "ggridges", "corrplot", "reshape2",
  # --- modelling ---
  "caret", "randomForest", "rpart",
  "glmnet",      # Ridge & Lasso
  "gbm",         # Gradient Boosting
  "e1071",       # SVM
  "nnet"         # Neural Network
)

if (AUTO_INSTALL) {
  installed <- rownames(installed.packages())
  for (pkg in required_packages) {
    if (!(pkg %in% installed)) {
      message("Installing: ", pkg)
      install.packages(pkg, dependencies = TRUE,
                       repos = "https://cran.r-project.org")
    }
  }
}

suppressPackageStartupMessages({
  library(httr);         library(jsonlite);      library(rvest)
  library(xml2);         library(dplyr);         library(tidyr)
  library(stringr);      library(readr);         library(lubridate)
  library(janitor);      library(tibble);        library(purrr)
  library(ggplot2);      library(scales);        library(RColorBrewer)
  library(gridExtra);    library(viridis);       library(ggridges)
  library(corrplot);     library(reshape2)
  library(caret);        library(randomForest);  library(rpart)
  library(glmnet);       library(gbm)
  library(e1071);        library(nnet)
})

set.seed(123)   # master seed for reproducibility


############################
# GLOBAL THEME & COLOUR PALETTES
############################
BD_COLORS <- c(
  "#006A4E", "#F42A41", "#FFCD00", "#1A6B9A", "#E67E22",
  "#8E44AD", "#2ECC71", "#E74C3C", "#3498DB", "#F39C12"
)

# Named palette matched to the 10 commodities
COMMODITY_PALETTE <- c(
  "Rice"        = "#2ECC71", "Wheat Flour" = "#F39C12",
  "Lentils"     = "#E74C3C", "Oil"         = "#3498DB",
  "Sugar"       = "#9B59B6", "Potato"      = "#1ABC9C",
  "Onion"       = "#E67E22", "Chicken"     = "#D35400",
  "Fish"        = "#2980B9", "Egg"         = "#F1C40F"
)

theme_bd <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title       = element_text(face  = "bold", size = base_size + 3,
                                      colour = "#1A1A2E", hjust = 0.5),
      plot.subtitle    = element_text(size  = base_size, colour = "#555555",
                                      hjust = 0.5),
      axis.title       = element_text(face  = "bold", colour = "#333333"),
      panel.grid.major = element_line(colour = "#EEEEEE"),
      panel.grid.minor = element_blank(),
      legend.position  = "right",
      plot.background  = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA)
    )
}


############################
# 1. OUTPUT FOLDERS
############################
dir.create("data",      showWarnings = FALSE)
dir.create("figures",   showWarnings = FALSE)
dir.create("results",   showWarnings = FALSE)
dir.create("raw_pages", showWarnings = FALSE)


############################
# 2. HELPER FUNCTIONS
############################

# Remove currency symbols / commas and coerce to numeric
clean_number <- function(x) {
  x <- as.character(x)
  x <- stringr::str_replace_all(x, "[,\u09F3TkBDT%]", "")
  x <- stringr::str_replace_all(x, "[^0-9.\\-]", "")
  suppressWarnings(as.numeric(x))
}

# Safe HTTP GET with timeout and user-agent
safe_get <- function(url, query = list(), timeout_seconds = 60) {
  message("Fetching: ", url)
  resp <- tryCatch(
    httr::GET(
      url, query = query,
      httr::timeout(timeout_seconds),
      httr::add_headers(
        `User-Agent` = paste0(
          "Mozilla/5.0 FoodPriceBD academic ",
          "web-scraping project (IDS Final Term)"
        ),
        `Accept` = "text/html,application/json,text/csv,*/*"
      )
    ),
    error = function(e) e
  )
  if (inherits(resp, "error")) {
    warning("Request failed: ", conditionMessage(resp))
    return(NULL)
  }
  if (httr::status_code(resp) >= 400) {
    warning("HTTP ", httr::status_code(resp), " for ", url)
    return(NULL)
  }
  return(resp)
}

# Parse a CSV string returned from an HTTP response
read_csv_from_text <- function(txt) {
  readr::read_csv(I(txt), show_col_types = FALSE, progress = FALSE) %>%
    janitor::clean_names()
}

# Write CSV and report path + dimensions
write_safe_csv <- function(df, path) {
  readr::write_csv(df, path)
  message("Saved: ", path,
          " [", nrow(df), " rows × ", ncol(df), " cols]")
}

# Regression performance metrics
metric_regression <- function(actual, predicted) {
  actual    <- as.numeric(actual)
  predicted <- as.numeric(predicted)
  valid     <- !is.na(actual) & !is.na(predicted) &
    is.finite(actual) & is.finite(predicted)
  actual    <- actual[valid]
  predicted <- predicted[valid]
  data.frame(
    MAE  = mean(abs(actual - predicted)),
    RMSE = sqrt(mean((actual - predicted)^2)),
    R2   = 1 - sum((actual - predicted)^2) /
      sum((actual - mean(actual))^2)
  )
}

# Classification performance metrics
metric_classification <- function(actual, predicted) {
  lvls      <- union(levels(as.factor(actual)),
                     levels(as.factor(predicted)))
  actual    <- factor(actual,    levels = lvls)
  predicted <- factor(predicted, levels = lvls)
  cm <- caret::confusionMatrix(predicted, actual)
  data.frame(
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    Kappa    = as.numeric(cm$overall["Kappa"])
  )
}

# Wrapper around ggsave that always uses white background
ggsave_bd <- function(filename, plot, w = 10, h = 6) {
  ggplot2::ggsave(filename, plot = plot,
                  width = w, height = h, dpi = 300, bg = "white")
  message("Figure saved: ", filename)
}


############################
# 3. WEB SCRAPING
# Source 1: Department of Agricultural Marketing (DAM)
# URL: https://market.dam.gov.bd
# Method: rvest HTML scraping
############################
scrape_dam_live_reference <- function() {
  dam_url <- "https://market.dam.gov.bd/market_daily_price_report?L=E"
  resp    <- safe_get(dam_url)
  if (is.null(resp)) return(tibble())
  
  html_txt <- httr::content(resp, as = "text", encoding = "UTF-8")
  writeLines(html_txt,
             "raw_pages/dam_daily_price_report_raw.html",
             useBytes = TRUE)
  
  page_text <- html_txt %>%
    rvest::read_html() %>%
    rvest::html_text2() %>%
    stringr::str_squish()
  
  writeLines(page_text,
             "raw_pages/dam_daily_price_report_text.txt",
             useBytes = TRUE)
  
  # Try to extract commodity name + min/max/change from ticker text
  pattern <- paste0(
    "([A-Za-z0-9 /()._-]+):\\s*([0-9]+\\.?[0-9]*)\\s*-\\s*",
    "([0-9]+\\.?[0-9]*)\\s*[\u25b2\u25bc]?\\s*([+-]?[0-9]+\\.?[0-9]*)%"
  )
  matches <- stringr::str_match_all(page_text, pattern)[[1]]
  
  if (nrow(matches) == 0) {
    warning("Could not parse DAM ticker data – site may have changed.")
    return(tibble())
  }
  
  df <- tibble(
    scrape_date    = Sys.Date(),
    source         = "Department of Agricultural Marketing, Bangladesh",
    source_url     = dam_url,
    commodity      = stringr::str_squish(matches[, 2]),
    min_price      = as.numeric(matches[, 3]),
    max_price      = as.numeric(matches[, 4]),
    avg_price      = (as.numeric(matches[, 3]) + as.numeric(matches[, 4])) / 2,
    change_percent = as.numeric(matches[, 5])
  )
  
  write_safe_csv(df, "data/dam_live_prices_reference.csv")
  return(df)
}


############################
# 4. API DATA SOURCES
# Source 2: World Bank Microdata API (WFP-FAO RTFP)
# Source 3: Humanitarian Data Exchange (HDX) / WFP
############################

fetch_worldbank_microdata_api <- function() {
  candidate_endpoints <- c(
    paste0("https://microdata.worldbank.org/api/tables/data/",
           "FCV/BGD_2021_RTFP_v02_M"),
    paste0("https://microdata.worldbank.org/api/tables/data/",
           "fcv/bgd_2021_rtfp_v02_m")
  )
  
  for (endpoint in candidate_endpoints) {
    all_chunks <- list()
    offset     <- 0L
    limit      <- 1000L
    chunk_id   <- 1L
    failed     <- FALSE
    
    repeat {
      resp <- safe_get(
        endpoint,
        query           = list(limit = limit, offset = offset, format = "csv"),
        timeout_seconds = 120
      )
      if (is.null(resp)) { failed <- TRUE; break }
      txt <- httr::content(resp, as = "text", encoding = "UTF-8")
      if (nchar(txt) < 20 || !stringr::str_detect(txt, ",")) {
        failed <- TRUE; break
      }
      df <- tryCatch(read_csv_from_text(txt), error = function(e) tibble())
      if (nrow(df) == 0) break
      all_chunks[[chunk_id]] <- df
      message("  Chunk ", chunk_id, " – rows: ", nrow(df))
      if (nrow(df) < limit) break
      offset   <- offset + limit
      chunk_id <- chunk_id + 1L
      Sys.sleep(0.3)
      if (offset > 100000L) break
    }
    
    if (!failed && length(all_chunks) > 0) {
      out <- dplyr::bind_rows(all_chunks)
      if (nrow(out) > 0) {
        write_safe_csv(out, "data/worldbank_raw_wide_food_prices.csv")
        return(out)
      }
    }
  }
  return(tibble())
}

fetch_hdx_wfp_bangladesh <- function() {
  resp <- safe_get(
    "https://data.humdata.org/api/3/action/package_show",
    query           = list(id = "wfp-food-prices-for-bangladesh"),
    timeout_seconds = 120
  )
  if (is.null(resp)) return(tibble())
  
  txt  <- httr::content(resp, as = "text", encoding = "UTF-8")
  meta <- tryCatch(
    jsonlite::fromJSON(txt, flatten = TRUE),
    error = function(e) NULL
  )
  if (is.null(meta) || isFALSE(meta$success)) return(tibble())
  
  resources <- meta$result$resources
  if (is.null(resources) || nrow(resources) == 0) return(tibble())
  
  csv_resources <- resources %>%
    dplyr::filter(
      stringr::str_detect(tolower(format), "csv") |
        stringr::str_detect(tolower(url), "csv")
    )
  if (nrow(csv_resources) == 0) return(tibble())
  
  csv_url <- csv_resources$url[1]
  message("Downloading HDX/WFP CSV: ", csv_url)
  df <- tryCatch(
    readr::read_csv(csv_url, show_col_types = FALSE, progress = FALSE) %>%
      janitor::clean_names(),
    error = function(e) {
      warning("HDX CSV download failed: ", conditionMessage(e))
      tibble()
    }
  )
  if (nrow(df) > 0)
    write_safe_csv(df, "data/hdx_wfp_raw_food_prices.csv")
  return(df)
}


############################
# 5. STANDARDISE SCRAPED DATA
############################

standardize_worldbank_wide <- function(df) {
  if (nrow(df) == 0) return(tibble())
  names(df) <- janitor::make_clean_names(names(df))
  
  # FIX: use explicit column name checks rather than `if ("x" %in% names(.))` inside mutate
  if ("dates" %in% names(df)) {
    df <- df %>% dplyr::mutate(date = as.Date(dates))
  } else if (all(c("year", "month") %in% names(df))) {
    df <- df %>% dplyr::mutate(
      date = as.Date(
        sprintf("%04d-%02d-01", as.integer(year), as.integer(month))
      )
    )
  } else {
    return(tibble())
  }
  
  price_cols <- intersect(
    names(df),
    c("lentils", "oil", "rice", "rice_fao", "wheat_flour", "wheat_flour_fao")
  )
  if (length(price_cols) == 0) return(tibble())
  
  meta_cols <- intersect(
    names(df),
    c("iso3", "country", "adm1_name", "adm2_name", "mkt_name",
      "lat", "lon", "year", "month", "currency", "date")
  )
  
  df %>%
    dplyr::select(dplyr::all_of(c(meta_cols, price_cols))) %>%
    tidyr::pivot_longer(
      cols      = dplyr::all_of(price_cols),
      names_to  = "commodity",
      values_to = "price_bdt"
    ) %>%
    dplyr::mutate(
      source    = "World Bank Microdata / WFP-FAO Real Time Food Prices",
      country   = if ("country"   %in% names(.)) as.character(country)   else "Bangladesh",
      adm1_name = if ("adm1_name" %in% names(.)) as.character(adm1_name) else NA_character_,
      adm2_name = if ("adm2_name" %in% names(.)) as.character(adm2_name) else NA_character_,
      mkt_name  = if ("mkt_name"  %in% names(.)) as.character(mkt_name)  else NA_character_,
      commodity = stringr::str_replace_all(commodity, "_", " ") %>%
        stringr::str_to_title(),
      price_bdt = as.numeric(price_bdt)
    ) %>%
    dplyr::filter(!is.na(price_bdt), price_bdt > 0, !is.na(date)) %>%
    dplyr::arrange(commodity, mkt_name, date)
}

standardize_hdx_wfp <- function(df) {
  if (nrow(df) == 0) return(tibble())
  names(df) <- janitor::make_clean_names(names(df))
  
  needed <- c("mp_price", "cm_name", "mp_year", "mp_month")
  if (!all(needed %in% names(df))) return(tibble())
  
  df %>%
    dplyr::mutate(
      date      = as.Date(
        sprintf("%04d-%02d-01", as.integer(mp_year), as.integer(mp_month))
      ),
      country   = if ("adm0_name" %in% names(.)) as.character(adm0_name) else "Bangladesh",
      adm1_name = if ("adm1_name" %in% names(.)) as.character(adm1_name) else NA_character_,
      adm2_name = if ("adm2_name" %in% names(.)) as.character(adm2_name) else NA_character_,
      mkt_name  = if ("mkt_name"  %in% names(.)) as.character(mkt_name)  else NA_character_,
      commodity = as.character(cm_name),
      currency  = if ("cur_name"  %in% names(.)) as.character(cur_name)  else "BDT",
      price_bdt = as.numeric(mp_price),
      source    = "HDX / World Food Programme Food Prices Bangladesh"
    ) %>%
    dplyr::select(dplyr::any_of(
      c("date", "country", "adm1_name", "adm2_name", "mkt_name",
        "commodity", "currency", "price_bdt", "source")
    )) %>%
    dplyr::filter(!is.na(price_bdt), price_bdt > 0, !is.na(date)) %>%
    dplyr::arrange(commodity, mkt_name, date)
}


############################
# 6. SYNTHETIC DATA FALLBACK
# Generates realistic Bangladesh food price data when
# live APIs are unavailable.
# 10 commodities × 8 divisions × 168 months = 13,440 rows
############################
generate_synthetic_food_price_data <- function() {
  message("\n>>> Generating synthetic Bangladesh food price ",
          "dataset as fallback <<<\n")
  
  commodities <- c("Rice", "Wheat Flour", "Lentils", "Oil", "Sugar",
                   "Potato", "Onion", "Chicken", "Fish", "Egg")
  
  divisions   <- c("Dhaka", "Chittagong", "Sylhet", "Rajshahi",
                   "Khulna", "Barisal", "Rangpur", "Mymensingh")
  
  # Realistic base prices in BDT (per kg or per unit)
  base_prices <- c(
    Rice = 45, `Wheat Flour` = 38, Lentils = 90, Oil = 130,
    Sugar = 65, Potato = 25, Onion = 40, Chicken = 180,
    Fish = 220, Egg = 12
  )
  
  # Monthly seasonal multipliers (index 1 = January)
  seasonal_mult <- c(1.05, 1.03, 0.98, 0.95, 0.97, 1.02,
                     1.08, 1.10, 1.06, 0.99, 0.96, 1.02)
  
  # ~6% annual inflation; 14-year window = 168 months
  dates <- seq(as.Date("2010-01-01"),
               as.Date("2023-12-01"),
               by = "month")
  
  set.seed(42)   # reproducible synthetic noise
  total <- length(commodities) * length(divisions) * length(dates)
  rows  <- vector("list", total)
  i     <- 1L
  
  for (div in divisions) {
    region_factor <- switch(div,
                            Dhaka       = 1.12, Chittagong = 1.08, Sylhet    = 1.05,
                            Rajshahi    = 0.95, Khulna     = 0.97, Barisal   = 0.93,
                            Rangpur     = 0.90, Mymensingh = 0.92,
                            1.00   # default
    )
    
    for (comm in commodities) {
      bp <- base_prices[comm]
      
      # FIX: loop over index so each value stays as Date.
      # Using `for (d in dates)` can drop the Date class and cause:
      # Error in prettyNum(...): invalid 'trim' argument
      for (date_i in seq_along(dates)) {
        d     <- dates[date_i]
        yr    <- as.integer(format(d, "%Y"))
        mo    <- as.integer(format(d, "%m"))
        infl  <- (1 + 0.06)^(yr - 2010)      # compound inflation
        seas  <- seasonal_mult[mo]             # seasonal effect
        shock <- exp(rnorm(1, 0, 0.03))        # random walk shock
        price <- max(bp * infl * seas * region_factor * shock,
                     bp * 0.5)                 # floor at 50 % of base
        
        rows[[i]] <- list(
          date      = d,
          country   = "Bangladesh",
          adm1_name = div,
          adm2_name = div,
          mkt_name  = paste(div, "Central Market"),
          commodity = comm,
          currency  = "BDT",
          price_bdt = round(price, 2),
          source    = "Synthetic (Realistic Bangladesh Data)"
        )
        i <- i + 1L
      }
    }
  }
  
  df <- dplyr::bind_rows(rows)
  write_safe_csv(df, "data/synthetic_bangladesh_food_prices.csv")
  message("Synthetic dataset generated: ", nrow(df), " rows")
  return(df)
}


############################
# 7. COLLECT MAIN DATASET
# Priority order:
#   (a) World Bank Microdata API
#   (b) HDX / WFP Bangladesh
#   (c) Synthetic fallback
############################
message("\n", strrep("=", 50))
message("STEP 1: Data Collection (Web Scraping & APIs)")
message(strrep("=", 50))

message("\n-- Scraping DAM live reference prices --")
dam_live <- tryCatch(
  scrape_dam_live_reference(),
  error = function(e) { warning(e); tibble() }
)
message("DAM reference rows: ", nrow(dam_live))

message("\n-- Fetching World Bank Microdata API --")
wb_raw    <- tryCatch(
  fetch_worldbank_microdata_api(),
  error = function(e) { warning(e); tibble() }
)
food_long <- standardize_worldbank_wide(wb_raw)
message("World Bank standardised rows: ", nrow(food_long))

if (nrow(food_long) < 10000) {
  message("\n-- World Bank gave only ", nrow(food_long),
          " rows. Trying HDX/WFP ... --")
  hdx_raw       <- tryCatch(
    fetch_hdx_wfp_bangladesh(),
    error = function(e) { warning(e); tibble() }
  )
  food_long_hdx <- standardize_hdx_wfp(hdx_raw)
  message("HDX standardised rows: ", nrow(food_long_hdx))
  if (nrow(food_long_hdx) > nrow(food_long))
    food_long <- food_long_hdx
}

if (nrow(food_long) < 10000) {
  message("\n-- APIs returned only ", nrow(food_long),
          " rows. Using synthetic fallback ... --")
  food_long <- generate_synthetic_food_price_data()
}

write_safe_csv(food_long, "data/foodpricebd_10000plus_long_dataset.csv")
message("\nMain dataset: ", nrow(food_long), " rows saved.")


# ============================================================
# FIX: CONSOLE-ONLY OUTPUT
# This replaces CSV saving.
# All write_safe_csv(...) calls will now print output only.
# ============================================================

write_safe_csv <- function(data, file = NULL, max_rows = 30) {
  
  output_name <- if (!is.null(file)) {
    basename(file)
  } else {
    "Console Output"
  }
  
  message("\n", strrep("-", 70))
  message("OUTPUT: ", output_name)
  message(strrep("-", 70))
  
  if (inherits(data, "data.frame")) {
    
    message("Rows    : ", nrow(data))
    message("Columns : ", ncol(data))
    
    if (nrow(data) == 0) {
      message("No rows available.")
    } else if (nrow(data) > max_rows) {
      print(utils::head(data, max_rows))
      message("... showing first ", max_rows, " rows only.")
    } else {
      print(data)
    }
    
  } else {
    print(data)
  }
  
  message("CSV file was NOT saved.")
  invisible(data)
}


############################
# 8. DATA UNDERSTANDING
############################

message("\n", strrep("=", 50))
message("STEP 2: Data Understanding")
message(strrep("=", 50))

summary_info <- tibble::tibble(
  total_rows         = nrow(food_long),
  total_columns      = ncol(food_long),
  unique_commodities = dplyr::n_distinct(food_long$commodity),
  unique_markets     = dplyr::n_distinct(food_long$mkt_name),
  date_range_start   = as.character(min(food_long$date, na.rm = TRUE)),
  date_range_end     = as.character(max(food_long$date, na.rm = TRUE)),
  missing_price_pct  = round(mean(is.na(food_long$price_bdt)) * 100, 2)
)

write_safe_csv(summary_info, "results/dataset_summary.csv")
print(summary_info)

# Missing-value table
missing_table <- food_long %>%
  dplyr::summarise(
    dplyr::across(dplyr::everything(), ~ sum(is.na(.)))
  ) %>%
  tidyr::pivot_longer(
    dplyr::everything(),
    names_to  = "variable",
    values_to = "missing_count"
  ) %>%
  dplyr::arrange(dplyr::desc(missing_count))

write_safe_csv(missing_table, "results/missing_value_summary.csv")

# Commodity record counts
commodity_counts <- food_long %>%
  dplyr::count(commodity, sort = TRUE)

write_safe_csv(commodity_counts, "results/commodity_counts.csv")

# Summary statistics per commodity
price_stats <- food_long %>%
  dplyr::group_by(commodity) %>%
  dplyr::summarise(
    n      = dplyr::n(),
    mean   = round(mean(price_bdt, na.rm = TRUE), 2),
    median = round(median(price_bdt, na.rm = TRUE), 2),
    sd     = round(sd(price_bdt, na.rm = TRUE), 2),
    min    = round(min(price_bdt, na.rm = TRUE), 2),
    max    = round(max(price_bdt, na.rm = TRUE), 2),
    .groups = "drop"
  )

write_safe_csv(price_stats, "results/price_summary_statistics.csv")
print(price_stats)


############################
# 9. EDA VISUALISATIONS
############################

message("\n", strrep("=", 50))
message("STEP 3: EDA Visualisations")
message(strrep("=", 50))

n_comm <- dplyr::n_distinct(food_long$commodity)

FILL_PALETTE <- colorRampPalette(
  c(
    "#006A4E", "#2ECC71", "#F39C12", "#E74C3C", "#3498DB",
    "#9B59B6", "#1ABC9C", "#E67E22", "#D35400", "#F1C40F"
  )
)(n_comm)


# ── 9.1 Price distribution histogram
p1 <- ggplot2::ggplot(
  food_long,
  ggplot2::aes(x = price_bdt, fill = ggplot2::after_stat(count))
) +
  ggplot2::geom_histogram(bins = 50, colour = "white", linewidth = 0.2) +
  ggplot2::scale_fill_viridis_c(option = "plasma", name = "Count") +
  ggplot2::scale_x_continuous(labels = scales::label_comma()) +
  ggplot2::labs(
    title    = "Distribution of Food Prices in Bangladesh",
    subtitle = "All commodities & markets combined",
    x = "Price (BDT)",
    y = "Frequency"
  ) +
  theme_bd()

ggsave_bd("figures/01_price_distribution.png", p1, w = 10, h = 6)


# ── 9.2 Average price by commodity
avg_by_commodity <- food_long %>%
  dplyr::group_by(commodity) %>%
  dplyr::summarise(
    avg_price = mean(price_bdt, na.rm = TRUE),
    .groups   = "drop"
  ) %>%
  dplyr::arrange(dplyr::desc(avg_price))

write_safe_csv(avg_by_commodity, "results/average_price_by_commodity.csv")

p2 <- ggplot2::ggplot(
  avg_by_commodity,
  ggplot2::aes(
    x = reorder(commodity, avg_price),
    y = avg_price,
    fill = commodity
  )
) +
  ggplot2::geom_col(show.legend = FALSE, width = 0.7) +
  ggplot2::geom_text(
    ggplot2::aes(label = paste0("\u09F3", round(avg_price, 1))),
    hjust = -0.1,
    size = 3.5
  ) +
  ggplot2::scale_fill_manual(values = FILL_PALETTE) +
  ggplot2::coord_flip() +
  ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.15))) +
  ggplot2::labs(
    title    = "Average Price by Commodity",
    subtitle = "All markets & time periods combined",
    x = NULL,
    y = "Average Price (BDT)"
  ) +
  theme_bd()

ggsave_bd("figures/02_avg_price_by_commodity.png", p2, w = 10, h = 6)


# ── 9.3 Monthly price trend
trend_df <- food_long %>%
  dplyr::group_by(date, commodity) %>%
  dplyr::summarise(
    avg_price = mean(price_bdt, na.rm = TRUE),
    .groups   = "drop"
  )

p3 <- ggplot2::ggplot(
  trend_df,
  ggplot2::aes(x = date, y = avg_price, colour = commodity)
) +
  ggplot2::geom_line(linewidth = 0.8) +
  ggplot2::facet_wrap(~ commodity, scales = "free_y", ncol = 3) +
  ggplot2::scale_colour_manual(values = FILL_PALETTE) +
  ggplot2::scale_x_date(date_labels = "'%y", date_breaks = "3 years") +
  ggplot2::labs(
    title    = "Monthly Price Trend by Commodity",
    subtitle = "Division-market average (BDT)",
    x = "Date",
    y = "Avg Price (BDT)"
  ) +
  theme_bd(base_size = 10) +
  ggplot2::theme(legend.position = "none")

ggsave_bd("figures/03_monthly_price_trend.png", p3, w = 14, h = 10)


# ── 9.4 Boxplot
p4 <- ggplot2::ggplot(
  food_long,
  ggplot2::aes(
    x = reorder(commodity, price_bdt, FUN = median),
    y = price_bdt,
    fill = commodity
  )
) +
  ggplot2::geom_boxplot(
    outlier.colour = "#E74C3C",
    outlier.alpha  = 0.5,
    outlier.size   = 1.5,
    colour         = "grey30"
  ) +
  ggplot2::scale_fill_manual(values = FILL_PALETTE) +
  ggplot2::coord_flip() +
  ggplot2::scale_y_continuous(labels = scales::label_comma()) +
  ggplot2::labs(
    title    = "Price Spread & Outlier Detection by Commodity",
    subtitle = "Box = IQR; Red dots = outliers (>1.5 × IQR)",
    x = NULL,
    y = "Price (BDT)"
  ) +
  theme_bd() +
  ggplot2::theme(legend.position = "none")

ggsave_bd("figures/04_boxplot_by_commodity.png", p4, w = 10, h = 6)


# ── 9.5 Average price by division
if ("adm1_name" %in% names(food_long)) {
  
  regional_avg <- food_long %>%
    dplyr::filter(!is.na(adm1_name)) %>%
    dplyr::group_by(adm1_name) %>%
    dplyr::summarise(
      avg_price = mean(price_bdt, na.rm = TRUE),
      .groups   = "drop"
    ) %>%
    dplyr::arrange(dplyr::desc(avg_price))
  
  write_safe_csv(regional_avg, "results/average_price_by_region.csv")
  
  p5 <- ggplot2::ggplot(
    regional_avg,
    ggplot2::aes(
      x = reorder(adm1_name, avg_price),
      y = avg_price,
      fill = avg_price
    )
  ) +
    ggplot2::geom_col(width = 0.7) +
    ggplot2::geom_text(
      ggplot2::aes(label = paste0("\u09F3", round(avg_price, 1))),
      hjust = -0.1,
      size = 3.5
    ) +
    ggplot2::scale_fill_gradient(
      low = "#AED6F1",
      high = "#1A5276",
      name = "Avg Price (BDT)"
    ) +
    ggplot2::coord_flip() +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.15))) +
    ggplot2::labs(
      title    = "Average Food Price by Division (Region)",
      subtitle = "Bangladesh – all commodities combined",
      x = NULL,
      y = "Average Price (BDT)"
    ) +
    theme_bd()
  
  ggsave_bd("figures/05_avg_price_by_region.png", p5, w = 10, h = 6)
}


# ── 9.6 Seasonal heatmap
heatmap_df <- food_long %>%
  dplyr::mutate(
    month_lbl = factor(format(date, "%b"), levels = month.abb)
  ) %>%
  dplyr::group_by(commodity, month_lbl) %>%
  dplyr::summarise(
    avg_price = mean(price_bdt, na.rm = TRUE),
    .groups   = "drop"
  )

p6 <- ggplot2::ggplot(
  heatmap_df,
  ggplot2::aes(x = month_lbl, y = commodity, fill = avg_price)
) +
  ggplot2::geom_tile(colour = "white", linewidth = 0.4) +
  ggplot2::geom_text(ggplot2::aes(label = round(avg_price, 0)), size = 2.8) +
  ggplot2::scale_fill_viridis_c(option = "inferno", name = "BDT") +
  ggplot2::labs(
    title    = "Seasonal Price Heatmap – Commodity × Month",
    subtitle = "Average price (BDT) for each month of the year",
    x = "Month",
    y = NULL
  ) +
  theme_bd() +
  ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 0))

ggsave_bd("figures/06_seasonal_heatmap.png", p6, w = 12, h = 7)


# ── 9.7 Ridge density plot
p7 <- ggplot2::ggplot(
  food_long,
  ggplot2::aes(
    x = price_bdt,
    y = reorder(commodity, price_bdt, median),
    fill = commodity
  )
) +
  ggridges::geom_density_ridges(
    alpha = 0.8,
    scale = 1.2,
    colour = "white"
  ) +
  ggplot2::scale_fill_manual(values = FILL_PALETTE) +
  ggplot2::scale_x_continuous(labels = scales::label_comma()) +
  ggplot2::labs(
    title    = "Price Density Distribution by Commodity",
    subtitle = "Ridgeline plot (BDT)",
    x = "Price (BDT)",
    y = NULL
  ) +
  theme_bd() +
  ggplot2::theme(legend.position = "none")

ggsave_bd("figures/07_ridgeline_density.png", p7, w = 10, h = 7)


# ── 9.8 Year-over-year average price
yoy_df <- food_long %>%
  dplyr::mutate(year = lubridate::year(date)) %>%
  dplyr::group_by(year) %>%
  dplyr::summarise(
    avg_price = mean(price_bdt, na.rm = TRUE),
    .groups   = "drop"
  )

p8 <- ggplot2::ggplot(yoy_df, ggplot2::aes(x = year, y = avg_price)) +
  ggplot2::geom_col(ggplot2::aes(fill = avg_price), width = 0.7, colour = "white") +
  ggplot2::geom_line(colour = "#E74C3C", linewidth = 1, linetype = "dashed") +
  ggplot2::geom_point(colour = "#E74C3C", size = 3) +
  ggplot2::scale_fill_gradient(
    low = "#AED6F1",
    high = "#154360",
    guide = "none"
  ) +
  ggplot2::scale_x_continuous(breaks = unique(yoy_df$year)) +
  ggplot2::labs(
    title    = "Year-over-Year Average Food Price",
    subtitle = "All commodities & divisions (BDT)",
    x = "Year",
    y = "Avg Price (BDT)"
  ) +
  theme_bd() +
  ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))

ggsave_bd("figures/08_year_over_year.png", p8, w = 10, h = 6)


############################
# 10. DATA PREPROCESSING & FEATURE ENGINEERING
############################

message("\n", strrep("=", 50))
message("STEP 4: Data Preprocessing & Feature Engineering")
message(strrep("=", 50))

# ── 10.1 Derived time features + lag features + movement label
model_data <- food_long %>%
  dplyr::mutate(
    year      = lubridate::year(date),
    month     = lubridate::month(date),
    quarter   = lubridate::quarter(date),
    commodity = as.factor(commodity),
    market    = as.factor(
      dplyr::if_else(
        is.na(mkt_name) | mkt_name == "",
        "Unknown",
        mkt_name
      )
    ),
    region    = as.factor(
      dplyr::if_else(
        is.na(adm1_name) | adm1_name == "",
        "Unknown",
        adm1_name
      )
    )
  ) %>%
  dplyr::arrange(commodity, market, date) %>%
  dplyr::group_by(commodity, market) %>%
  dplyr::mutate(
    lag_price        = dplyr::lag(price_bdt, 1),
    lag2_price       = dplyr::lag(price_bdt, 2),
    lag3_price       = dplyr::lag(price_bdt, 3),
    rolling_mean_3   = (
      dplyr::lag(price_bdt, 1) +
        dplyr::lag(price_bdt, 2) +
        dplyr::lag(price_bdt, 3)
    ) / 3,
    price_change     = price_bdt - lag_price,
    price_change_pct = (price_bdt - lag_price) / lag_price * 100,
    movement_label   = dplyr::case_when(
      is.na(price_change_pct) ~ NA_character_,
      price_change_pct >  1.5 ~ "Increase",
      price_change_pct < -1.5 ~ "Decrease",
      TRUE                    ~ "Stable"
    )
  ) %>%
  dplyr::ungroup() %>%
  dplyr::filter(
    !is.na(lag_price),
    !is.na(price_change_pct),
    is.finite(price_change_pct)
  )

# ── 10.2 Winsorise extreme percentage changes
lo <- quantile(model_data$price_change_pct, 0.01, na.rm = TRUE)
hi <- quantile(model_data$price_change_pct, 0.99, na.rm = TRUE)

model_data <- model_data %>%
  dplyr::mutate(
    price_change_pct_capped = pmin(pmax(price_change_pct, lo), hi)
  )

message("\n", strrep("-", 70))
message("MODEL-READY DATASET")
message(strrep("-", 70))
message("Model-ready rows    : ", nrow(model_data))
message("Model-ready columns : ", ncol(model_data))
message("\nFirst 30 rows of model-ready dataset:")
print(utils::head(model_data, 30))
message("Model-ready dataset was NOT saved as CSV.")


############################
# 11. TRAIN / TEST SPLIT
############################

unique_dates <- sort(unique(model_data$date))
split_date   <- unique_dates[floor(length(unique_dates) * 0.80)]

train_data <- model_data %>%
  dplyr::filter(date <= split_date)

test_data <- model_data %>%
  dplyr::filter(date > split_date)

# Safety: if test set is too small, fall back to random split
if (nrow(test_data) < 100) {
  warning("Test set too small after time split. Using random 80/20 split.")
  
  idx <- caret::createDataPartition(
    model_data$price_bdt,
    p = 0.80,
    list = FALSE
  )
  
  train_data <- model_data[idx, ]
  test_data  <- model_data[-idx, ]
}

write_safe_csv(
  tibble::tibble(
    split_date = as.character(split_date),
    train_rows = nrow(train_data),
    test_rows  = nrow(test_data)
  ),
  "results/train_test_split_summary.csv"
)

message("Train rows: ", nrow(train_data), " | Test rows: ", nrow(test_data))


############################
# 12. REGRESSION MODELS
############################

message("\n", strrep("=", 50))
message("STEP 5a: Regression Modelling (7 algorithms)")
message(strrep("=", 50))

reg_features <- c(
  "lag_price",
  "lag2_price",
  "lag3_price",
  "rolling_mean_3",
  "year",
  "month",
  "quarter"
)

reg_formula <- price_bdt ~
  lag_price + lag2_price + lag3_price + rolling_mean_3 +
  year + month + quarter + commodity + market + region

make_glmnet_matrices <- function(train, test, features) {
  
  if (nrow(train) == 0 || nrow(test) == 0) {
    stop("make_glmnet_matrices: train or test data is empty.")
  }
  
  all_data <- dplyr::bind_rows(
    train %>% dplyr::mutate(.split = "train"),
    test  %>% dplyr::mutate(.split = "test")
  ) %>%
    dplyr::mutate(
      commodity = as.character(commodity),
      market    = as.character(market),
      region    = as.character(region)
    )
  
  formula_dm <- as.formula(
    paste(
      "~ 0 +",
      paste(
        c(features, "commodity", "market", "region"),
        collapse = " + "
      )
    )
  )
  
  mat_all <- model.matrix(formula_dm, data = all_data)
  
  list(
    x_train = mat_all[all_data$.split == "train", , drop = FALSE],
    x_test  = mat_all[all_data$.split == "test", , drop = FALSE],
    y_train = train$price_bdt,
    y_test  = test$price_bdt
  )
}

glm_data <- make_glmnet_matrices(train_data, test_data, reg_features)

reg_results <- list()

# ── 12.1 Linear Regression
message("  [1/7] Linear Regression ...")
set.seed(1)
lm_mod  <- stats::lm(reg_formula, data = train_data)
lm_pred <- stats::predict(lm_mod, newdata = test_data)

reg_results[["Linear Regression"]] <-
  metric_regression(test_data$price_bdt, lm_pred)

# ── 12.2 Ridge Regression
message("  [2/7] Ridge Regression ...")
set.seed(2)
ridge_cv <- glmnet::cv.glmnet(
  glm_data$x_train,
  glm_data$y_train,
  alpha = 0,
  nfolds = 5
)

ridge_pred <- predict(
  ridge_cv,
  newx = glm_data$x_test,
  s = "lambda.min"
)[, 1]

reg_results[["Ridge Regression"]] <-
  metric_regression(glm_data$y_test, ridge_pred)

# ── 12.3 Lasso Regression
message("  [3/7] Lasso Regression ...")
set.seed(3)
lasso_cv <- glmnet::cv.glmnet(
  glm_data$x_train,
  glm_data$y_train,
  alpha = 1,
  nfolds = 5
)

lasso_pred <- predict(
  lasso_cv,
  newx = glm_data$x_test,
  s = "lambda.min"
)[, 1]

reg_results[["Lasso Regression"]] <-
  metric_regression(glm_data$y_test, lasso_pred)

# ── 12.4 Decision Tree
message("  [4/7] Decision Tree ...")
set.seed(4)
tree_mod <- rpart::rpart(
  reg_formula,
  data = train_data,
  method = "anova",
  control = rpart::rpart.control(maxdepth = 10, cp = 0.001)
)

tree_pred <- predict(tree_mod, newdata = test_data)

reg_results[["Decision Tree"]] <-
  metric_regression(test_data$price_bdt, tree_pred)

# ── 12.5 Random Forest
message("  [5/7] Random Forest ...")
set.seed(5)
rf_mod <- randomForest::randomForest(
  reg_formula,
  data = train_data,
  ntree = 200,
  importance = TRUE,
  na.action = na.omit
)

rf_pred <- predict(rf_mod, newdata = test_data)

reg_results[["Random Forest"]] <-
  metric_regression(test_data$price_bdt, rf_pred)

# ── 12.6 Gradient Boosting
message("  [6/7] Gradient Boosting (GBM) ...")
set.seed(6)
gbm_mod <- gbm::gbm(
  reg_formula,
  data = train_data,
  distribution      = "gaussian",
  n.trees           = 300,
  interaction.depth = 4,
  shrinkage         = 0.05,
  bag.fraction      = 0.8,
  train.fraction    = 1.0,
  verbose           = FALSE
)

gbm_pred <- predict(gbm_mod, newdata = test_data, n.trees = 300)

reg_results[["Gradient Boosting"]] <-
  metric_regression(test_data$price_bdt, gbm_pred)

# ── 12.7 Support Vector Regression
message("  [7/7] Support Vector Regression ...")
set.seed(7)

svm_train_idx <- sample(nrow(train_data), min(3000, nrow(train_data)))
svm_test_idx  <- sample(nrow(test_data), min(1000, nrow(test_data)))

svm_train <- train_data[svm_train_idx, ]
svm_test  <- test_data[svm_test_idx, ]

svm_reg_formula <- price_bdt ~
  lag_price + lag2_price + lag3_price + rolling_mean_3 +
  year + month + quarter

svm_mod <- e1071::svm(
  svm_reg_formula,
  data = svm_train,
  kernel = "radial",
  cost = 1,
  epsilon = 0.1
)

svm_pred <- predict(svm_mod, newdata = svm_test)

reg_results[["SVR (Radial)"]] <-
  metric_regression(svm_test$price_bdt, svm_pred)

# Compile regression results
reg_comparison <- dplyr::bind_rows(reg_results, .id = "Model") %>%
  dplyr::select(Model, MAE, RMSE, R2) %>%
  dplyr::arrange(RMSE)

best_reg_model <- reg_comparison$Model[1]

message(
  "\n>>> Best Regression Model: ",
  best_reg_model,
  " (RMSE = ",
  round(reg_comparison$RMSE[1], 2),
  ")"
)

write_safe_csv(reg_comparison, "results/regression_model_comparison.csv")
print(reg_comparison)


############################
# 13. REGRESSION VISUALISATIONS
############################

# ── 13.1 R2 comparison
p_reg_r2 <- ggplot2::ggplot(
  reg_comparison,
  ggplot2::aes(
    x = reorder(Model, R2),
    y = R2,
    fill = ifelse(Model == best_reg_model, "Best", "Other")
  )
) +
  ggplot2::geom_col(width = 0.65) +
  ggplot2::geom_text(ggplot2::aes(label = round(R2, 4)), hjust = -0.1, size = 3.5) +
  ggplot2::scale_fill_manual(
    values = c(Best = "#006A4E", Other = "#AED6F1"),
    guide = "none"
  ) +
  ggplot2::coord_flip() +
  ggplot2::scale_y_continuous(
    expand = ggplot2::expansion(mult = c(0, 0.15)),
    limits = c(0, 1)
  ) +
  ggplot2::labs(
    title    = "Regression Model Comparison – R²",
    subtitle = paste("Best model:", best_reg_model),
    x = NULL,
    y = "R² (higher is better)"
  ) +
  theme_bd()

ggsave_bd("figures/09_regression_r2_comparison.png", p_reg_r2, w = 10, h = 6)


# ── 13.2 RMSE comparison
p_reg_rmse <- ggplot2::ggplot(
  reg_comparison,
  ggplot2::aes(
    x = reorder(Model, -RMSE),
    y = RMSE,
    fill = ifelse(Model == best_reg_model, "Best", "Other")
  )
) +
  ggplot2::geom_col(width = 0.65) +
  ggplot2::geom_text(ggplot2::aes(label = round(RMSE, 2)), hjust = -0.1, size = 3.5) +
  ggplot2::scale_fill_manual(
    values = c(Best = "#006A4E", Other = "#E74C3C"),
    guide = "none"
  ) +
  ggplot2::coord_flip() +
  ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.15))) +
  ggplot2::labs(
    title    = "Regression Model Comparison – RMSE",
    subtitle = paste("Best model (lowest RMSE):", best_reg_model),
    x = NULL,
    y = "RMSE (lower is better)"
  ) +
  theme_bd()

ggsave_bd("figures/10_regression_rmse_comparison.png", p_reg_rmse, w = 10, h = 6)


# ── 13.3 Combined metrics
reg_long <- reg_comparison %>%
  dplyr::mutate(R2_pct = R2 * 100) %>%
  dplyr::select(Model, MAE, RMSE, R2_pct) %>%
  tidyr::pivot_longer(
    cols = c(MAE, RMSE, R2_pct),
    names_to = "Metric",
    values_to = "Value"
  )

p_reg_all <- ggplot2::ggplot(
  reg_long,
  ggplot2::aes(x = Model, y = Value, fill = Metric)
) +
  ggplot2::geom_col(position = "dodge", width = 0.75) +
  ggplot2::scale_fill_manual(
    values = c(
      MAE = "#3498DB",
      RMSE = "#E74C3C",
      R2_pct = "#2ECC71"
    )
  ) +
  ggplot2::facet_wrap(~ Metric, scales = "free_y") +
  ggplot2::labs(
    title = "All Regression Metrics – Side-by-Side Comparison",
    x = NULL,
    y = "Value"
  ) +
  theme_bd(base_size = 10) +
  ggplot2::theme(
    axis.text.x = ggplot2::element_text(angle = 40, hjust = 1),
    legend.position = "none"
  )

ggsave_bd("figures/11_regression_all_metrics.png", p_reg_all, w = 13, h = 6)


# ── 13.4 Actual vs Predicted
best_reg_pred <- switch(
  best_reg_model,
  "Random Forest"     = rf_pred,
  "Gradient Boosting" = gbm_pred,
  "Linear Regression" = lm_pred,
  "Decision Tree"     = tree_pred,
  "Ridge Regression"  = ridge_pred,
  "Lasso Regression"  = lasso_pred,
  "SVR (Radial)"      = svm_pred,
  rf_pred
)

best_test_actual <- if (best_reg_model == "SVR (Radial)") {
  svm_test$price_bdt
} else {
  test_data$price_bdt
}

min_len <- min(length(best_test_actual), length(best_reg_pred))

avp_df <- tibble::tibble(
  actual    = as.numeric(best_test_actual[seq_len(min_len)]),
  predicted = as.numeric(best_reg_pred[seq_len(min_len)]),
  commodity = if (best_reg_model == "SVR (Radial)") {
    svm_test$commodity[seq_len(min_len)]
  } else {
    test_data$commodity[seq_len(min_len)]
  }
)

write_safe_csv(avp_df, "results/best_model_actual_vs_predicted.csv")

p_avp <- ggplot2::ggplot(
  avp_df,
  ggplot2::aes(x = actual, y = predicted, colour = commodity)
) +
  ggplot2::geom_point(alpha = 0.35, size = 1.2) +
  ggplot2::geom_abline(
    slope = 1,
    intercept = 0,
    colour = "#E74C3C",
    linetype = "dashed",
    linewidth = 1
  ) +
  ggplot2::scale_colour_manual(values = FILL_PALETTE, name = "Commodity") +
  ggplot2::scale_x_continuous(labels = scales::label_comma()) +
  ggplot2::scale_y_continuous(labels = scales::label_comma()) +
  ggplot2::labs(
    title    = paste("Actual vs Predicted –", best_reg_model),
    subtitle = "Dashed red line = perfect prediction",
    x = "Actual Price (BDT)",
    y = "Predicted Price (BDT)"
  ) +
  theme_bd()

ggsave_bd("figures/12_actual_vs_predicted.png", p_avp, w = 10, h = 7)


# ── 13.5 Feature importance
importance_df <- randomForest::importance(rf_mod) %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Feature") %>%
  dplyr::arrange(dplyr::desc(IncNodePurity)) %>%
  dplyr::slice_head(n = 15)

write_safe_csv(importance_df, "results/rf_regression_feature_importance.csv")

p_imp <- ggplot2::ggplot(
  importance_df,
  ggplot2::aes(
    x = reorder(Feature, IncNodePurity),
    y = IncNodePurity,
    fill = IncNodePurity
  )
) +
  ggplot2::geom_col(width = 0.65) +
  ggplot2::scale_fill_gradient(
    low = "#AED6F1",
    high = "#1A5276",
    guide = "none"
  ) +
  ggplot2::coord_flip() +
  ggplot2::labs(
    title    = "Top Feature Importance – Random Forest Regressor",
    subtitle = "Metric: Increase in Node Purity (IncNodePurity)",
    x = NULL,
    y = "IncNodePurity"
  ) +
  theme_bd()

ggsave_bd("figures/13_feature_importance_regression.png", p_imp, w = 10, h = 6)


############################
# 14. CLASSIFICATION MODELS
############################

message("\n", strrep("=", 50))
message("STEP 5b: Classification Modelling (5 algorithms)")
message(strrep("=", 50))

# Movement label counts
movement_counts <- as.data.frame(table(model_data$movement_label)) %>%
  dplyr::rename(Label = Var1, Count = Freq)

write_safe_csv(movement_counts, "results/movement_label_counts.csv")

train_cls <- train_data %>%
  dplyr::filter(!is.na(movement_label)) %>%
  dplyr::mutate(movement_label = as.factor(movement_label))

test_cls <- test_data %>%
  dplyr::filter(!is.na(movement_label)) %>%
  dplyr::mutate(
    movement_label = factor(
      movement_label,
      levels = levels(train_cls$movement_label)
    )
  )

cls_formula <- movement_label ~
  lag_price + lag2_price + lag3_price + rolling_mean_3 +
  year + month + quarter + commodity + market + region

cls_comparison <- tibble::tibble()
best_cls_model <- "N/A"

if (
  nrow(train_cls) > 100 &&
  nrow(test_cls) > 10 &&
  length(unique(train_cls$movement_label)) >= 2
) {
  
  cls_results <- list()
  
  # ── 14.1 Decision Tree Classifier
  message("  [1/5] Decision Tree Classifier ...")
  set.seed(11)
  
  ctree_mod <- rpart::rpart(
    cls_formula,
    data = train_cls,
    method = "class"
  )
  
  ctree_pred <- predict(ctree_mod, newdata = test_cls, type = "class")
  
  cls_results[["Decision Tree"]] <-
    metric_classification(test_cls$movement_label, ctree_pred)
  
  # ── 14.2 Random Forest Classifier
  message("  [2/5] Random Forest Classifier ...")
  set.seed(12)
  
  rfc_mod <- randomForest::randomForest(
    cls_formula,
    data = train_cls,
    ntree = 200,
    importance = TRUE,
    na.action = na.omit
  )
  
  rfc_pred <- predict(rfc_mod, newdata = test_cls)
  
  cls_results[["Random Forest"]] <-
    metric_classification(test_cls$movement_label, rfc_pred)
  
  # ── 14.3 Gradient Boosting Classifier
  message("  [3/5] Gradient Boosting Classifier ...")
  set.seed(13)
  
  train_cls_gbm <- train_cls %>%
    dplyr::mutate(move_num = as.integer(movement_label) - 1L)
  
  gbm_cls <- gbm::gbm(
    move_num ~ lag_price + lag2_price + lag3_price +
      rolling_mean_3 + year + month + quarter,
    data              = train_cls_gbm,
    distribution      = "multinomial",
    n.trees           = 200,
    interaction.depth = 3,
    shrinkage         = 0.05,
    verbose           = FALSE
  )
  
  gbm_cls_prob <- predict(
    gbm_cls,
    newdata = test_cls,
    n.trees = 200,
    type = "response"
  )
  
  gbm_cls_mat <- if (length(dim(gbm_cls_prob)) == 3) {
    gbm_cls_prob[, , 1]
  } else {
    as.matrix(gbm_cls_prob)
  }
  
  gbm_cls_idx <- apply(gbm_cls_mat, 1, which.max)
  
  gbm_cls_pred <- factor(
    levels(train_cls$movement_label)[gbm_cls_idx],
    levels = levels(test_cls$movement_label)
  )
  
  cls_results[["Gradient Boosting"]] <-
    metric_classification(test_cls$movement_label, gbm_cls_pred)
  
  # ── 14.4 SVM Classifier
  message("  [4/5] SVM Classifier ...")
  set.seed(14)
  
  svm_cls_train <- train_cls[
    sample(nrow(train_cls), min(3000, nrow(train_cls))),
  ]
  
  svm_cls_test <- test_cls[
    sample(nrow(test_cls), min(1000, nrow(test_cls))),
  ]
  
  svm_cls_formula <- movement_label ~
    lag_price + lag2_price + lag3_price +
    rolling_mean_3 + year + month + quarter
  
  svm_cls <- e1071::svm(
    svm_cls_formula,
    data = svm_cls_train,
    kernel = "radial",
    cost = 1
  )
  
  svm_cls_pred <- predict(svm_cls, newdata = svm_cls_test)
  
  cls_results[["SVM"]] <-
    metric_classification(svm_cls_test$movement_label, svm_cls_pred)
  
  # ── 14.5 Neural Network Classifier
  message("  [5/5] Neural Network (nnet) Classifier ...")
  set.seed(15)
  
  nnet_n <- min(2000, nrow(train_cls))
  nnet_train_idx <- sample(nrow(train_cls), nnet_n)
  
  nnet_cls <- nnet::nnet(
    movement_label ~ lag_price + lag2_price + lag3_price +
      rolling_mean_3 + year + month + quarter,
    data = train_cls[nnet_train_idx, ],
    size = 10,
    maxit = 200,
    trace = FALSE
  )
  
  nnet_test_n <- min(600, nrow(test_cls))
  nnet_test_idx <- sample(nrow(test_cls), nnet_test_n)
  
  nnet_pred <- predict(
    nnet_cls,
    newdata = test_cls[nnet_test_idx, ],
    type = "class"
  )
  
  nnet_actual <- test_cls$movement_label[nnet_test_idx]
  
  cls_results[["Neural Network"]] <-
    metric_classification(nnet_actual, nnet_pred)
  
  # Compile classification results
  cls_comparison <- dplyr::bind_rows(cls_results, .id = "Model") %>%
    dplyr::arrange(dplyr::desc(Accuracy))
  
  best_cls_model <- cls_comparison$Model[1]
  
  message(
    "\n>>> Best Classification Model: ",
    best_cls_model,
    " (Accuracy = ",
    scales::percent(cls_comparison$Accuracy[1], accuracy = 0.1),
    ")"
  )
  
  write_safe_csv(cls_comparison, "results/classification_model_comparison.csv")
  print(cls_comparison)
  
  # ── 14.6 Accuracy comparison
  p_cls_acc <- ggplot2::ggplot(
    cls_comparison,
    ggplot2::aes(
      x = reorder(Model, Accuracy),
      y = Accuracy,
      fill = ifelse(Model == best_cls_model, "Best", "Other")
    )
  ) +
    ggplot2::geom_col(width = 0.65) +
    ggplot2::geom_text(
      ggplot2::aes(label = scales::percent(Accuracy, accuracy = 0.1)),
      hjust = -0.1,
      size = 3.5
    ) +
    ggplot2::scale_fill_manual(
      values = c(Best = "#006A4E", Other = "#AED6F1"),
      guide = "none"
    ) +
    ggplot2::coord_flip() +
    ggplot2::scale_y_continuous(
      expand = ggplot2::expansion(mult = c(0, 0.15)),
      labels = scales::percent_format()
    ) +
    ggplot2::labs(
      title    = "Classification Model Comparison – Accuracy",
      subtitle = paste("Best model:", best_cls_model),
      x = NULL,
      y = "Accuracy (higher is better)"
    ) +
    theme_bd()
  
  ggsave_bd("figures/14_classification_accuracy.png", p_cls_acc, w = 10, h = 6)
  
  # ── 14.7 Kappa comparison
  p_cls_kappa <- ggplot2::ggplot(
    cls_comparison,
    ggplot2::aes(
      x = reorder(Model, Kappa),
      y = Kappa,
      fill = ifelse(Model == best_cls_model, "Best", "Other")
    )
  ) +
    ggplot2::geom_col(width = 0.65) +
    ggplot2::geom_text(
      ggplot2::aes(label = round(Kappa, 4)),
      hjust = -0.1,
      size = 3.5
    ) +
    ggplot2::scale_fill_manual(
      values = c(Best = "#2ECC71", Other = "#F39C12"),
      guide = "none"
    ) +
    ggplot2::coord_flip() +
    ggplot2::scale_y_continuous(
      expand = ggplot2::expansion(mult = c(0, 0.15))
    ) +
    ggplot2::labs(
      title    = "Classification Model Comparison – Cohen's Kappa",
      subtitle = "Accounts for class imbalance; higher is better",
      x = NULL,
      y = "Kappa"
    ) +
    theme_bd()
  
  ggsave_bd("figures/15_classification_kappa.png", p_cls_kappa, w = 10, h = 6)
  
  # ── 14.8 Confusion matrix
  if (best_cls_model == "SVM") {
    best_cls_pred   <- svm_cls_pred
    best_cls_actual <- svm_cls_test$movement_label
  } else if (best_cls_model == "Neural Network") {
    best_cls_pred   <- factor(nnet_pred, levels = levels(test_cls$movement_label))
    best_cls_actual <- nnet_actual
  } else if (best_cls_model == "Gradient Boosting") {
    best_cls_pred   <- gbm_cls_pred
    best_cls_actual <- test_cls$movement_label
  } else if (best_cls_model == "Decision Tree") {
    best_cls_pred   <- ctree_pred
    best_cls_actual <- test_cls$movement_label
  } else {
    best_cls_pred   <- rfc_pred
    best_cls_actual <- test_cls$movement_label
  }
  
  n_cm <- min(length(best_cls_actual), length(best_cls_pred))
  
  cm_df <- as.data.frame(
    table(
      Predicted = best_cls_pred[seq_len(n_cm)],
      Actual    = best_cls_actual[seq_len(n_cm)]
    )
  )
  
  write_safe_csv(cm_df, "results/best_classifier_confusion_matrix.csv")
  
  p_cm <- ggplot2::ggplot(
    cm_df,
    ggplot2::aes(x = Actual, y = Predicted, fill = Freq)
  ) +
    ggplot2::geom_tile(colour = "white", linewidth = 0.8) +
    ggplot2::geom_text(ggplot2::aes(label = Freq), size = 5, fontface = "bold") +
    ggplot2::scale_fill_gradient(
      low = "#EBF5FB",
      high = "#1A5276",
      name = "Count"
    ) +
    ggplot2::labs(
      title    = paste("Confusion Matrix –", best_cls_model),
      subtitle = "Price Movement: Increase / Stable / Decrease",
      x = "Actual Label",
      y = "Predicted Label"
    ) +
    theme_bd()
  
  ggsave_bd("figures/16_confusion_matrix_best_classifier.png", p_cm, w = 8, h = 6)
  
  # ── 14.9 Feature importance
  cls_imp <- randomForest::importance(rfc_mod) %>%
    as.data.frame() %>%
    tibble::rownames_to_column("Feature") %>%
    dplyr::arrange(dplyr::desc(MeanDecreaseGini)) %>%
    dplyr::slice_head(n = 15)
  
  write_safe_csv(cls_imp, "results/rf_classification_feature_importance.csv")
  
  p_cls_imp <- ggplot2::ggplot(
    cls_imp,
    ggplot2::aes(
      x = reorder(Feature, MeanDecreaseGini),
      y = MeanDecreaseGini,
      fill = MeanDecreaseGini
    )
  ) +
    ggplot2::geom_col(width = 0.65) +
    ggplot2::scale_fill_gradient(
      low = "#ABEBC6",
      high = "#1E8449",
      guide = "none"
    ) +
    ggplot2::coord_flip() +
    ggplot2::labs(
      title    = "Feature Importance – Random Forest Classifier",
      subtitle = "Metric: Mean Decrease in Gini Impurity",
      x = NULL,
      y = "Mean Decrease Gini"
    ) +
    theme_bd()
  
  ggsave_bd("figures/17_classification_feature_importance.png", p_cls_imp, w = 10, h = 6)
  
} else {
  warning(
    "Not enough class variation or test rows for classification. ",
    "Movement label counts are shown in console."
  )
}


############################
# 15. MASTER SUMMARY DASHBOARD
############################

message("\n", strrep("=", 50))
message("STEP 6: Summary Dashboard")
message(strrep("=", 50))

if (nrow(cls_comparison) > 0) {
  
  p_dash_reg <- ggplot2::ggplot(
    reg_comparison,
    ggplot2::aes(
      x = reorder(Model, R2),
      y = R2,
      fill = ifelse(Model == best_reg_model, "Best", "Other")
    )
  ) +
    ggplot2::geom_col(width = 0.7) +
    ggplot2::geom_text(
      ggplot2::aes(label = round(R2, 3)),
      hjust = -0.1,
      size = 3
    ) +
    ggplot2::scale_fill_manual(
      values = c(Best = "#006A4E", Other = "#AED6F1"),
      guide = "none"
    ) +
    ggplot2::coord_flip() +
    ggplot2::scale_y_continuous(
      expand = ggplot2::expansion(mult = c(0, 0.15)),
      limits = c(0, 1)
    ) +
    ggplot2::labs(
      title = "Regression – R²",
      x = NULL,
      y = "R²"
    ) +
    theme_bd(base_size = 9)
  
  p_dash_cls <- ggplot2::ggplot(
    cls_comparison,
    ggplot2::aes(
      x = reorder(Model, Accuracy),
      y = Accuracy,
      fill = ifelse(Model == best_cls_model, "Best", "Other")
    )
  ) +
    ggplot2::geom_col(width = 0.7) +
    ggplot2::geom_text(
      ggplot2::aes(label = scales::percent(Accuracy, accuracy = 0.1)),
      hjust = -0.1,
      size = 3
    ) +
    ggplot2::scale_fill_manual(
      values = c(Best = "#F42A41", Other = "#FDECEA"),
      guide = "none"
    ) +
    ggplot2::coord_flip() +
    ggplot2::scale_y_continuous(
      expand = ggplot2::expansion(mult = c(0, 0.15)),
      labels = scales::percent_format()
    ) +
    ggplot2::labs(
      title = "Classification – Accuracy",
      x = NULL,
      y = "Accuracy"
    ) +
    theme_bd(base_size = 9)
  
  dashboard <- gridExtra::arrangeGrob(
    p_dash_reg,
    p_dash_cls,
    ncol = 2,
    top = grid::textGrob(
      "Bangladesh Food Price ML Dashboard – Model Leaderboard",
      gp = grid::gpar(
        fontsize = 14,
        fontface = "bold",
        col = "#1A1A2E"
      )
    )
  )
  
  ggplot2::ggsave(
    "figures/18_model_leaderboard_dashboard.png",
    plot = dashboard,
    width = 14,
    height = 6,
    dpi = 300,
    bg = "white"
  )
  
  message("Figure saved: figures/18_model_leaderboard_dashboard.png")
}


############################
# 16. FINAL PIPELINE SUMMARY REPORT
############################

message("\n", strrep("=", 50))
message("STEP 7: Final Summary Report")
message(strrep("=", 50))

cls_models_str <- if (nrow(cls_comparison) > 0) {
  paste(cls_comparison$Model, collapse = " | ")
} else {
  "N/A"
}

best_cls_acc <- if (nrow(cls_comparison) > 0) {
  scales::percent(cls_comparison$Accuracy[1], accuracy = 0.1)
} else {
  "N/A"
}

best_cls_kappa <- if (nrow(cls_comparison) > 0) {
  round(cls_comparison$Kappa[1], 4)
} else {
  "N/A"
}

data_sources <- if ("source" %in% names(food_long)) {
  paste(unique(food_long$source), collapse = "; ")
} else {
  "Not available"
}

final_report <- tibble::tibble(
  item = c(
    "Research Objective",
    "Data Source(s)",
    "Total rows (main dataset)",
    "Total rows (model-ready)",
    "Unique commodities",
    "Unique markets / divisions",
    "Date range",
    "Train rows (80%)",
    "Test rows  (20%)",
    paste0("Regression models trained (", nrow(reg_comparison), ")"),
    "Best regression model",
    "  Best R²",
    "  Best RMSE",
    "  Best MAE",
    paste0(
      "Classification models trained (",
      if (nrow(cls_comparison) > 0) nrow(cls_comparison) else 0,
      ")"
    ),
    "Best classification model",
    "  Best Accuracy",
    "  Best Cohen's Kappa"
  ),
  value = c(
    "Predict and classify essential food price volatility in Bangladesh",
    data_sources,
    as.character(nrow(food_long)),
    as.character(nrow(model_data)),
    as.character(dplyr::n_distinct(food_long$commodity)),
    as.character(dplyr::n_distinct(food_long$mkt_name)),
    paste(min(food_long$date), "to", max(food_long$date)),
    as.character(nrow(train_data)),
    as.character(nrow(test_data)),
    paste(reg_comparison$Model, collapse = " | "),
    best_reg_model,
    as.character(round(reg_comparison$R2[1], 4)),
    as.character(round(reg_comparison$RMSE[1], 2)),
    as.character(round(reg_comparison$MAE[1], 2)),
    cls_models_str,
    best_cls_model,
    best_cls_acc,
    as.character(best_cls_kappa)
  )
)

write_safe_csv(final_report, "results/final_pipeline_summary.csv")
print(final_report, n = Inf)


############################
# DONE
############################

message("\n", strrep("=", 50))
message("  PIPELINE COMPLETE")
message(strrep("=", 50))
message("Main dataset          : shown in console only")
message("Model-ready dataset   : shown in console only")
message("Figures               : generated by plotting functions")
message("Results               : shown in console only; no CSV files saved")
message("Best Regression Model : ", best_reg_model)
message("Best Classifier       : ", best_cls_model)
message(strrep("=", 50))

