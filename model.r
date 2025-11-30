# ==============================================================================
# MINIMALIST SCRIPT – Runs Models and Prints Summary
# Libraries: brms, dplyr, loo
# ==============================================================================

library(brms)
library(dplyr)
library(loo)
# No other plotting/data libraries needed.

# Set options and seed
options(mc.cores = parallel::detectCores())
set.seed(123)
options(future.globals.maxSize = 3 * 1024^3)

# ----------------------------- 1. LOAD & PREPARE DATA ------------------------
cat("Loading and preparing data...\n")
# *** IMPORTANT: UPDATE THIS PATH TO YOUR FILE LOCATION ***
df <- read.csv("C:\\Users\\ignac\\OneDrive - Aalto University\\Documents\\Aalto\\bayesian\\cleaned_data.csv",
               stringsAsFactors = FALSE)

df_clean <- df %>%
  filter(!Age.Group %in% "[All]", Sex %in% c("Male", "Female")) %>%
  group_by(Country, Sex, Age.Group) %>%
  summarise(mean_rate = mean(DeathRate, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    sex_num = ifelse(Sex == "Male", 1, 0),
    # Recoding Age Groups into a numeric factor (0 to 4)
    age_num = case_when(
      Age.Group %in% c("[0]", "[1-4]", "[5-9]", "[10-14]") ~ 0,
      Age.Group %in% c("[15-19]", "[20-24]", "[25-29]") ~ 1,
      Age.Group %in% c("[30-34]", "[35-39]", "[40-44]") ~ 2,
      Age.Group %in% c("[45-49]", "[50-54]", "[55-59]") ~ 3,
      Age.Group %in% c("[60-64]", "[65-69]", "[70-74]", "[75-79]", "[80-84]", "[85+]") ~ 4
    )
  ) %>% filter(!is.na(age_num))

df_clean$Country <- factor(df_clean$Country)
df_model <- df_clean %>% rename(y = mean_rate)

cat("N =", nrow(df_model), "| Countries =", n_distinct(df_model$Country), "\n\n")

# ----------------------------- 2. PRIORS & MODELS ---------------------------
prior_lognormal_fixed <- c(prior(normal(0,5), class="Intercept"), prior(normal(0,2), class="b"), prior(exponential(1), class="sigma"))
prior_lognormal_hier  <- c(prior_lognormal_fixed, prior(exponential(0.5), class="sd"))
prior_gamma <- c(prior(normal(0,5), class="Intercept"), prior(normal(0,2), class="b"),
                 prior(gamma(0.01,0.01), class="shape"), prior(exponential(0.5), class="sd"))

# Set parameters for the model fitting
n_chains <- 4
n_iter <- 4000
n_warmup_hier <- 2000
n_warmup_fixed <- 1000

cat("Fitting models (this may take a while with 4000 iterations)...\n")
# Model 1: Fixed effects, LogNormal
model1 <- brm(y ~ sex_num + age_num, data = df_model, family = lognormal(), prior = prior_lognormal_fixed,
              chains = n_chains, iter = n_iter, warmup = n_warmup_fixed, seed = 123, control = list(adapt_delta = 0.95), refresh = 0, save_pars = save_pars(all = TRUE))

# Model 2: Hierarchical (1|Country), LogNormal
model2 <- brm(y ~ sex_num + age_num + (1|Country), data = df_model, family = lognormal(), prior = prior_lognormal_hier,
              chains = n_chains, iter = n_iter, warmup = n_warmup_hier, seed = 123, control = list(adapt_delta = 0.95), refresh = 0, save_pars = save_pars(all = TRUE))

# Model 3: Hierarchical (1|Country), Gamma (log link)
model3 <- brm(y ~ sex_num + age_num + (1|Country), data = df_model, family = Gamma(link="log"), prior = prior_gamma,
              chains = n_chains, iter = n_iter, warmup = n_warmup_hier, seed = 123, control = list(adapt_delta = 0.95), refresh = 0, save_pars = save_pars(all = TRUE))

# ----------------------------- 3. LOO COMPARISON ---------------------------
cat("\n=== MODEL COMPARISON (PSIS-LOO) ===\n")
l1 <- loo(model1)
l2 <- loo(model2)
# reloo=TRUE is needed for the Gamma model to ensure robust results
l3 <- loo(model3, reloo = TRUE) 
print(l1)
print(l2)
print(l3)
comp <- loo_compare(l1, l2, l3)
rownames(comp) <- c("Model3_Gamma_Hier", "Model2_LogNormal_Hier", "Model1_LogNormal_Fixed")
print(round(comp, 1))

# ----------------------------- 4. PRINT ALL MODEL SUMMARIES ---------------------------
cat("\n\n#############################################\n")
cat("          1. SUMMARY FOR MODEL 1\n")
cat("#############################################\n")
print(summary(model1))

cat("\n\n#############################################\n")
cat("          2. SUMMARY FOR MODEL 2\n")
cat("#############################################\n")
print(summary(model2))

cat("\n\n#############################################\n")
cat("          3. SUMMARY FOR MODEL 3 (WINNER)\n")
cat("#############################################\n")
print(summary(model3))

cat("\n\n=== SCRIPT COMPLETE ===\n")
cat("Check the console output for the LOO comparison and model summaries.\n")
# ======================================================================
# 5. DIAGNOSTIC & FIT PLOTS
# ======================================================================
cat("\n\n=== GENERATING PLOTS ===\n")

library(ggplot2)
library(bayesplot)

# ----------------------------- 5.1 Traceplots for all models -----------------------------
cat("Plotting traceplots with bayesplot...\n")

# Extract posterior draws
posterior1 <- as.array(model1)
posterior2 <- as.array(model2)
posterior3 <- as.array(model3)

png("traceplot_model1.png", width=1400, height=800)
print(mcmc_trace(posterior1))
dev.off()

png("traceplot_model2.png", width=1400, height=800)
print(mcmc_trace(posterior2))
dev.off()

png("traceplot_model3.png", width=1400, height=800)
print(mcmc_trace(posterior3))
dev.off()

# ----------------------------- 5.2 Posterior predictive checks -----------------------------
cat("Plotting posterior predictive checks...\n")

# Basic PPC
png("pp_check_model1.png", width=1200, height=800)
print(pp_check(model1))
dev.off()

png("pp_check_model2.png", width=1200, height=800)
print(pp_check(model2))
dev.off()

png("pp_check_model3.png", width=1200, height=800)
print(pp_check(model3))
dev.off()

# Density overlay PPC
png("pp_check_overlay_model3.png", width=1200, height=800)
print(pp_check(model3, type = "dens_overlay"))
dev.off()


# ----------------------------- 5.3 Fitted vs Observed -----------------------------
cat("Plotting fitted vs observed...\n")

df_model$fitted_m3 <- fitted(model3)[,1]
df_model$pred_m3   <- posterior_predict(model3) %>% apply(2, median)

p1 <- ggplot(df_model, aes(x = fitted_m3, y = y)) +
  geom_point(alpha=0.5) +
  geom_abline(slope=1, intercept=0, color="red") +
  labs(title="Model 3: Fitted Values vs Observed", x="Fitted (mean)", y="Observed") +
  theme_minimal()

png("fitted_vs_observed_model3.png", width=1200, height=800)
print(p1)
dev.off()


# ----------------------------- 5.4 Country random effects (Model 3) -----------------------------
cat("Plotting random effects...\n")

re_m3 <- ranef(model3)$Country[,,1] %>% as.data.frame()
re_m3$Country <- rownames(ranef(model3)$Country)

p2 <- ggplot(re_m3, aes(x = reorder(Country, Estimate), y = Estimate)) +
  geom_point() +
  geom_errorbar(aes(ymin = Q2.5, ymax = Q97.5), width=0.2) +
  coord_flip() +
  labs(title="Model 3 Random Intercepts by Country",
       y="Random Intercept", x="Country") +
  theme_minimal()

png("country_random_effects_model3.png", width=1200, height=1600)
print(p2)
dev.off()


# ----------------------------- 5.5 Effect plots (sex & age) -----------------------------
cat("Plotting marginal effect estimates...\n")

png("marginal_effects_model3.png", width=1200, height=800)
print(marginal_effects(model3, surfaces = FALSE))
dev.off()

cat("=== PLOTS GENERATED ===\n")
cat("All plots saved to the working directory.\n")