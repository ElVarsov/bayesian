df_clean <- read.csv(
  "cleaned_data.csv",
  stringsAsFactors = FALSE
)

# -----------------------------
# 2. FILTER & AGGREGATE OVER YEARS -----------------------------------------
df_clean <- df_clean %>%
  filter(!Age.Group %in% "[All]", Sex %in% c("Male","Female")) %>%
  group_by(Country, Sex, Age.Group) %>%
  summarise(
    mean_rate = mean(DeathRate, na.rm = TRUE),
    var_rate  = var(DeathRate, na.rm = TRUE),
    n_years   = n(),
    .groups = "drop"
  )

cat("Columns detected after aggregation:\n")
print(colnames(df_clean))
cat("Total rows after aggregation:", nrow(df_clean), "\n")

# -----------------------------
# 3. CREATE MODEL VARIABLES -----------------------------------------------
df_clean <- df_clean %>%
  mutate(
    sex_num = ifelse(Sex == "Male", 1, 0),
    age_num = case_when(
      Age.Group %in% c("[0]", "[1-4]", "[5-9]", "[10-14]") ~ 0,
      Age.Group %in% c("[15-19]", "[20-24]", "[25-29]")   ~ 1,
      Age.Group %in% c("[30-34]", "[35-39]", "[40-44]")   ~ 2,
      Age.Group %in% c("[45-49]", "[50-54]", "[55-59]")   ~ 3,
      Age.Group %in% c("[60-64]", "[65-69]", "[70-74]", "[75-79]", "[80-84]", "[85+]") ~ 4
    ),
    log_mean_rate = log(mean_rate + 1e-6)  # avoid log(0)
  ) %>%
  filter(!is.na(age_num))  # remove unmatched age groups

# Make Country a factor for hierarchical model
df_clean$Country <- as.factor(df_clean$Country)

# Quick check
summary(df_clean$log_mean_rate)
summary(df_clean$sex_num)
summary(df_clean$age_num)
cat("Number of countries:", n_distinct(df_clean$Country), "\n")

# -----------------------------
# 4. BAYESIAN MODELS -------------------------------------------------------
set.seed(123)
options(mc.cores = parallel::detectCores())

# Priors
prior_simple <- c(
  prior(normal(0,5), class = "Intercept"),
  prior(normal(0,2), class = "b"),
  prior(exponential(1), class = "sigma")
)

prior_hier <- c(
  prior(normal(0,5), class = "Intercept"),
  prior(normal(0,2), class = "b"),
  prior(exponential(1), class = "sigma"),
  prior(exponential(0.5), class = "sd")
)

cat("\nFitting models (fixed + hierarchical by Country)...\n")

# Model 1: fixed effects only
model1 <- brm(
  log_mean_rate ~ sex_num + age_num,
  data = df_clean,
  family = gaussian(),
  prior = prior_simple,
  chains = 4, iter = 2000, warmup = 1000,
  seed = 123, refresh = 0,
  save_pars = save_pars(all = TRUE) 
)

# Model 2: hierarchical intercept by Country
model2 <- brm(
  log_mean_rate ~ sex_num + age_num + (1 | Country),
  data = df_clean,
  family = gaussian(),
  prior = prior_hier,
  chains = 4, iter = 2000, warmup = 1000,
  seed = 123, refresh = 0,
  save_pars = save_pars(all = TRUE) 
)

# -----------------------------
# 5. RESULTS ---------------------------------------------------------------
s1 <- summary(model1)
s2 <- summary(model2)

# LOO-CV (memory friendly)
l1 <- loo(model1, cores = 1, moment_match = TRUE)
l2 <- loo(model2, cores = 1, moment_match = TRUE)
comp <- loo_compare(l1, l2)

y <- df_clean$mean_rate
ppc1 <- posterior_predict(model1, ndraws = 300)
ppc2 <- posterior_predict(model2, ndraws = 300)

cat("\n", strrep("=", 85), "\n")
cat("       ALL NUMBERS YOU NEED — COPY FROM HERE INTO YOUR REPORT\n")
cat(strrep("=", 85), "\n\n")

cat("Final sample size                 :", nrow(df_clean), "\n")
cat("Number of countries               :", n_distinct(df_clean$Country), "\n\n")

cat("Convergence diagnostics\n")
cat("Model 1 – max Rhat               :", round(max(c(s1$fixed[,"Rhat"], s1$spec_pars[,"Rhat"])), 3), "\n")
cat("Model 1 – min Bulk ESS           :", round(min(c(s1$fixed[,"Bulk_ESS"], s1$spec_pars[,"Bulk_ESS"])), 0), "\n")
cat("Model 2 – max Rhat               :", round(max(c(s2$fixed[,"Rhat"], s2$spec_pars[,"Rhat"], s2$random$Country[,"Rhat"])), 3), "\n")
cat("Model 2 – min Bulk ESS           :", round(min(c(s2$fixed[,"Bulk_ESS"], s2$spec_pars[,"Bulk_ESS"], s2$random$Country[,"Bulk_ESS"])), 0), "\n\n")

cat("Fixed effects – Model 1 (posterior means)\n")
print(round(s1$fixed[,"Estimate"], 2))
cat("\nFixed effects – Model 2\n")
print(round(s2$fixed[,"Estimate"], 2))
cat("Country random intercepts (Model 2)\n")
print(round(s2$random$Country[,"Estimate"], 2))

cat("\nLOO-CV comparison\n")
cat("Model 1 ELPD =", round(l1$estimates[3,1], 1), " (SE =", round(l1$estimates[3,2], 1), ")\n")
cat("Model 2 ELPD =", round(l2$estimates[3,1], 1), " (SE =", round(l2$estimates[3,2], 1), ")\n")
cat("ELPD difference (Model2 better) =", round(comp[2,"elpd_diff"], 1),
    " ±", round(comp[2,"se_diff"], 1), "\n")

cat("\nPosterior predictive check (mean / sd of rate)\n")
cat("Observed      :", round(mean(y), 2), "/", round(sd(y), 2), "\n")
cat("Model 1 PPC   :", round(mean(colMeans(ppc1)), 2), "/", round(mean(apply(ppc1,1,sd)), 2), "\n")
cat("Model 2 PPC   :", round(mean(colMeans(ppc2)), 2), "/", round(mean(apply(ppc2,1,sd)), 2), "\n")

cat("\n", strrep("=", 85), "\n")
cat("COPY EVERYTHING ABOVE — YOU ARE NOW READY TO SUBMIT!\n")
cat(strrep("=", 85), "\n")