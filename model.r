# Load necessary libraries
library(brms)
library(dplyr)
library(tidyr)
library(loo)
library(rstan)
library(ggplot2)

# Set seed and parallel cores
set.seed(123)
options(mc.cores = parallel::detectCores())

# Define the small constant (epsilon) for fudging zeros
EPSILON <- 1e-6 

cat("\nUsing EPSILON =", EPSILON, "to fudge zeros for Gamma/Log-Normal models.\n")

# ==============================================================================
# 1-3. DATA PREPARATION
# ==============================================================================
# IMPORTANT: Adjust the file path to match your environment.
df_clean <- read.csv(
  "C:\\Users\\ignac\\OneDrive - Aalto University\\Documents\\Aalto\\bayesian\\cleaned_data.csv",
  stringsAsFactors = FALSE
)

# 2. FILTER & AGGREGATE OVER YEARS
df_clean <- df_clean %>%
  filter(!Age.Group %in% "[All]", Sex %in% c("Male","Female")) %>%
  group_by(Country, Sex, Age.Group) %>%
  summarise(
    mean_rate = mean(DeathRate, na.rm = TRUE),
    var_rate  = var(DeathRate, na.rm = TRUE),
    n_years   = n(),
    .groups = "drop"
  )

# 3. CREATE MODEL VARIABLES
df_clean <- df_clean %>%
  mutate(
    sex_num = ifelse(Sex == "Male", 1, 0),
    age_num = case_when(
      Age.Group %in% c("[0]", "[1-4]", "[5-9]", "[10-14]") ~ 0,
      Age.Group %in% c("[15-19]", "[20-24]", "[25-29]")    ~ 1,
      Age.Group %in% c("[30-34]", "[35-39]", "[40-44]")    ~ 2,
      Age.Group %in% c("[45-49]", "[50-54]", "[55-59]")    ~ 3,
      Age.Group %in% c("[60-64]", "[65-69]", "[70-74]", "[75-79]", "[80-84]", "[85+]") ~ 4
    )
  ) %>%
  filter(!is.na(age_num))

df_clean$Country <- as.factor(df_clean$Country)

# CRITICAL STEP: Create the Fudged Data Set
df_fudged <- df_clean %>%
  mutate(
    # FUDGE ZEROS for Gamma model: replaces 0 with EPSILON
    mean_rate_fudged = mean_rate + EPSILON,
    # Recalculate log_mean_rate using the fudged data
    log_mean_rate = log(mean_rate_fudged)
  )

# --- Priors ---
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

prior_gamma <- c(
  prior(normal(0, 5), class = "Intercept"),
  prior(normal(0, 2), class = "b"),
  prior(gamma(0.01, 0.01), class = "shape"),
  prior(exponential(0.5), class = "sd")
)

cat("\nFitting models on the fudged data set (N=", nrow(df_fudged), ")...\n")

# ==============================================================================
# 4. BAYESIAN MODELS (Refitted on FUDGED data)
# ==============================================================================

# --- Model 1: Gaussian Fixed effects (Log-scale) ---
model1 <- brm(
  log_mean_rate ~ sex_num + age_num,
  data = df_fudged, # <-- Use fudged data
  family = gaussian(),
  prior = prior_simple,
  chains = 4, iter = 2000, warmup = 1000,
  seed = 123, refresh = 0,
  save_pars = save_pars(all = TRUE)
)

# --- Model 2: Gaussian Hierarchical (Log-scale, Increased Iterations) ---
model2 <- brm(
  log_mean_rate ~ sex_num + age_num + (1 | Country),
  data = df_fudged, # <-- Use fudged data
  family = gaussian(),
  prior = prior_hier,
  chains = 4,
  iter = 4000,
  warmup = 2000,
  seed = 123, refresh = 0,
  save_pars = save_pars(all = TRUE)
)

# --- Model 3: Gamma Hierarchical (Raw FUDGED data) ---
model3 <- brm(
  mean_rate_fudged ~ sex_num + age_num + (1 | Country), # <-- Use fudged dependent variable
  data = df_fudged,
  family = Gamma(link = "log"),
  prior = prior_gamma,
  chains = 4,
  iter = 4000,
  warmup = 2000,
  seed = 123, refresh = 0,
  save_pars = save_pars(all = TRUE)
)

# ==============================================================================
# 5. RESULTS & LOO-CV (COMPROMISE COMPARISON)
# ==============================================================================
s1 <- summary(model1)
s2 <- summary(model2)
s3 <- summary(model3)
options(future.globals.maxSize = 800 * 1024^2)
# LOO-CV for Model 3 (Address Pareto k warning with reloo)
l3_reloo <- loo(model3, cores = 1, moment_match = TRUE, reloo = TRUE) 

# --- Fair Comparison on RAW SCALE (P(Y_raw | model)) ---
# We must use the RAW, unfudged data for the LL calculation!
Y_raw <- df_fudged$mean_rate 

# Log-Normal LL for Model 1 (We use dlnorm to estimate P(Y_raw | model1))
log_lik_M1_lognormal <- log_lik(
  model1, newdata = df_fudged, fun = function(data, draws) {
    # draws$mu and draws$sigma come from the log-scale Gaussian model
    ll <- dlnorm(Y_raw, meanlog = draws$mu, sdlog = draws$sigma, log = TRUE)
    return(ll)
  }
)
l1_lognormal <- loo(log_lik_M1_lognormal, cores = 1)

# Log-Normal LL for Model 2 
log_lik_M2_lognormal <- log_lik(
  model2, newdata = df_fudged, fun = function(data, draws) {
    ll <- dlnorm(Y_raw, meanlog = draws$mu, sdlog = draws$sigma, log = TRUE)
    return(ll)
  }
)
l2_lognormal <- loo(log_lik_M2_lognormal, cores = 1)


# Final Fair Comparison
comp_final <- loo_compare(l3_reloo, l2_lognormal, l1_lognormal)

# Posterior Predictive Checks (PPC)
ppc1 <- posterior_predict(model1, ndraws = 300)
ppc2 <- posterior_predict(model2, ndraws = 300)
ppc3 <- posterior_predict(model3, ndraws = 300)

cat("\n", strrep("=", 85), "\n")
cat("       FINAL RESULTS — ALL MODELS COMPARED ON RAW MEAN_RATE SCALE\n")
cat("       (Fudged data used for fitting, Raw data used for LL comparison)\n")
cat(strrep("=", 85), "\n\n")

# --- Sample Info ---
cat("Final sample size (Fudged Data Set) :", nrow(df_fudged), "\n")
cat("Count of Zero Death Rates fudged    :", sum(df_clean$mean_rate == 0), "\n")
cat("Number of countries                 :", n_distinct(df_fudged$Country), "\n\n")

# --- Convergence Diagnostics ---
cat("Convergence diagnostics (min Bulk ESS)\n")
cat("Model 1 – min Bulk ESS           :", round(min(c(s1$fixed[,"Bulk_ESS"], s1$spec_pars[,"Bulk_ESS"])), 0), "\n")
cat("Model 2 – min Bulk ESS           :", round(min(c(s2$fixed[,"Bulk_ESS"], s2$spec_pars[,"Bulk_ESS"], s2$random$Country[,"Bulk_ESS"])), 0), "\n")
cat("Model 3 – min Bulk ESS           :", round(min(c(s3$fixed[,"Bulk_ESS"], s3$spec_pars[,"Bulk_ESS"], s3$random$Country[,"Bulk_ESS"])), 0), "\n\n")

# --- Fixed Effects ---
cat("Fixed effects – Model 2 (Gaussian, log-scale posterior means)\n")
print(round(s2$fixed[,"Estimate"], 2))
cat("\nFixed effects – Model 3 (Gamma, log-scale posterior means)\n")
print(round(s3$fixed[,"Estimate"], 2))
cat("Country random intercepts SD (Model 3):\n")
print(round(s3$random$Country[,"Estimate"], 2))

# --- LOO-CV Comparison (FAIR) ---
cat("\nLOO-CV comparison (FAIR: All ELPDs calculated on RAW mean_rate scale)\n")
rownames(comp_final) <- c("Model3_Gamma", "Model2_LogNormal_Hier", "Model1_LogNormal_Fixed")
print(round(comp_final, 1))

# --- Posterior Predictive Check ---
cat("\nPosterior predictive check (mean / sd of rate)\n")
cat("Observed (RAW Data)     :", round(mean(Y_raw), 2), "/", round(sd(Y_raw), 2), "\n")
# Inverse-transform Models 1 and 2 PPCs to compare with raw scale
ppc1_raw <- exp(ppc1)
ppc2_raw <- exp(ppc2)

cat("Model 1 PPC (Log-Normal):", round(mean(colMeans(ppc1_raw)), 2), "/", round(mean(apply(ppc1_raw,1,sd)), 2), "\n")
cat("Model 2 PPC (Log-Normal):", round(mean(colMeans(ppc2_raw)), 2), "/", round(mean(apply(ppc2_raw,1,sd)), 2), "\n")
cat("Model 3 PPC (Gamma)     :", round(mean(colMeans(ppc3)), 2), "/", round(mean(apply(ppc3,1,sd)), 2), "\n")

cat("\n", strrep("=", 85), "\n")
cat("COPY EVERYTHING ABOVE — YOU ARE NOW READY TO SUBMIT!\n")
cat(strrep("=", 85), "\n")


# ==============================================================================
# 6. PLOTS
# ==============================================================================
message("\nGenerating Plots for the final report...")

# --- Plot 1: Model Fit Comparison (Observed vs. Model 3 Predicted Densities) ---
y_gamma_plot <- df_fudged$mean_rate_fudged
ndraws_ppc3 <- nrow(ppc3)
plot_data_ppc3_fixed <- data.frame(
  observed = rep(y_gamma_plot, times = ndraws_ppc3),
  predicted = as.vector(t(ppc3))
)

ggplot(plot_data_ppc3_fixed, aes(x = predicted)) +
  geom_density(aes(color = "Posterior Predictive"), alpha = 0.7, size = 1) +
  geom_density(aes(x = observed, color = "Observed (Fudged)"), size = 1, linetype = "dashed") +
  scale_color_manual(values = c("Posterior Predictive" = "darkblue", "Observed (Fudged)" = "darkred")) +
  labs(
    title = "Plot 1: Posterior Predictive Check - Observed (Fudged) vs. Model 3 (Gamma)",
    subtitle = paste0("Model 3 fit using mean_rate + ", EPSILON),
    x = "Mean Death Rate",
    y = "Density",
    color = "Data Type"
  ) +
  theme_minimal() +
  xlim(0, quantile(y_gamma_plot, 0.99) * 2) 
ggsave("plot_1_ppc_model3.png", width = 10, height = 6)
message("Plot 1 saved as 'plot_1_ppc_model3.png'")


# --- Plot 2: Convergence Check (Rhat for Model 3) ---
rhat_plot <- bayesplot::mcmc_rhat(bayesplot::rhat(model3)) +
  ggtitle("Plot 2: Rhat for Model 3 (Gamma)")
ggsave("plot_2_rhat_model3.png", width = 10, height = 8)
message("Plot 2 saved as 'plot_2_rhat_model3.png'")

# --- Plot 3: Model Comparison (LOO-CV Visualization) ---
loo_plot <- plot(comp_final, plot_type = "hpd") +
  ggtitle("Plot 3: LOO-CV Model Comparison (Raw Mean Rate Scale)")
ggsave("plot_3_loo_compare.png", width = 10, height = 6)
message("Plot 3 saved as 'plot_3_loo_compare.png'")


# --- Plot 4: Estimated Country Intercepts (Model 3) ---
library(tidybayes)
country_intercepts <- tidybayes::spread_draws(model3, r_Country[Country,]) %>%
  tidybayes::median_qi(.width = c(.95, .8)) %>%
  arrange(.value)

ggplot(country_intercepts, aes(x = .value, y = reorder(Country, .value))) +
  geom_pointinterval(aes(xmin = .lower, xmax = .upper), size = 0.5) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Plot 4: Estimated Country-Specific Intercepts (Model 3, log-scale)",
    subtitle = "Varying intercepts from the Gamma model fit to fudged data",
    x = "Log(Mean Death Rate) Intercept",
    y = "Country"
  ) +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 5),
    axis.title.y = element_blank()
  )
ggsave("plot_4_country_intercepts_model3.png", width = 12, height = 20)
message("Plot 4 saved as 'plot_4_country_intercepts_model3.png'")