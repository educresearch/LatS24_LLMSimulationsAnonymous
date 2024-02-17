# save charts to 850x650
# Loading necessary libraries for data manipulation and stats analysis
library(tidyverse)
library(stats)
library(scales)

# Read the CSV file
df_raw <- read.csv("./All.csv")


# Retaining only the specified columns
df_raw <- df_raw %>% select(model, temperature, prompt_number, SN,
                            congruent_ansCorrect, neutral_ansCorrect,
                            incongruent_ansCorrect)

# Renaming the columns
colnames(df_raw) <- c("model", "temperature", "prompt", "item",
                      "congruent", "neutral", "incongruent")


# Summarize the number of missing values in each column
missing_summary <- df_raw %>%
  summarise_all(~sum(is.na(.)))


# Print the summary
print(missing_summary)


# Install visdat package if you haven't already
# install.packages("visdat")

# Load visdat
library(visdat)

# Visualize missing data
vis_miss(df_raw)


# Recoding missing or null values to 0
df_raw[is.na(df_raw)] <- 0

# Recoding the values of prompt_number and SN
df_raw$prompt <- recode(df_raw$prompt, 
                        '1' = "a", '2' = "b", '3' = "c", '4' = "d")
df_raw$item <- recode(df_raw$item, 
                      '4' = "i", '5' = "ii", '6' = "iii", '7' = "iv", '8' = "v")


# Reshaping the data using melt (from reshape2 package)
df_melted <- df_raw %>% pivot_longer(
  cols = c(congruent, neutral, incongruent), 
  names_to = "spacing", 
  values_to = "correct")

mean_proportion <- df_melted %>% group_by(spacing, model, prompt) %>% 
  summarize(correct = mean(correct))




# Plotting 

# Calculate mean and standard error for each group
grouped_df_melted <- df_melted %>%
  group_by(model, spacing) %>%
  summarise(correct_mean = mean(correct, na.rm = TRUE),
            correct_se = sd(correct, na.rm = TRUE) / sqrt(n())) %>%
  ungroup()



# Plotting with numerical proportions above error bars + shift legend title
ggplot(grouped_df_melted, aes(x = model, y = correct_mean, fill = spacing)) +
  geom_bar(stat = "identity", position = position_dodge(), width = 0.7) +
  geom_errorbar(aes(ymin = correct_mean - correct_se, ymax = correct_mean + correct_se),
                width = 0.2, position = position_dodge(0.7)) +
  geom_text(aes(label = scales::percent(correct_mean, accuracy = 1),
                y = correct_mean + correct_se + 0.02), # Adjust '0.02' to ensure text is above error bars
            position = position_dodge(0.7),
            vjust = -0.5, # Adjust vertical position outside the bars
            size = 2.5) + # Smaller text size to reduce overlap and ensure clarity
  labs(title = "Spacing",
       x = "Model", y = "Proportion Correct", 
       fill = "") + # Remove title for the legend by setting it to an empty string
  scale_fill_brewer(palette = "Set1") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), # Make x-axis labels vertical
        legend.position = "top", # Move legend to the top
        plot.title = element_text(hjust = 0.5, face = "bold", size = 12, vjust = 1)) # Customize title appearance





# ANOVA 

anova_df_melted <- aov(correct ~ spacing + model + prompt + item, data = df_melted)

# Summary of ANOVA results

anova_summary_df_melted <- summary(anova_df_melted)

anova_summary_df_melted


# Logistic Regression

logit_model_df_melted <- glm(correct ~ spacing + model + prompt + item, 
                             family = binomial, data = df_melted)

# Summary of the logistic regression model

logreg_summary_df_melted <-summary(logit_model_df_melted)

logreg_summary_df_melted


# check using deviance for reduced model without spacing
logit_model_df_melted_reduced <- glm(correct ~ model + prompt + item, 
                                     family = binomial, data = df_melted)

df_melted_deviance_diff <- logit_model_df_melted_reduced$deviance - logit_model_df_melted$deviance
df_melted_chi_squared_test <- 1 - pchisq(df_melted_deviance_diff, df = 2)

# Print the p-value
print(df_melted_chi_squared_test)



#############################################################################################


df_melted_geminipro = df_melted[df_melted$'model' == 'GeminiPro',]
df_melted_gpt35_0613 = df_melted[df_melted$'model' == 'gpt-3.5-turbo-0613',]
df_melted_gpt35_1106 = df_melted[df_melted$'model' == 'gpt-3.5-turbo-1106',]
df_melted_gpt40_0613 = df_melted[df_melted$'model' == 'gpt-4-0613',]
df_melted_gpt40_1106 = df_melted[df_melted$'model' == 'gpt-4-1106-preview',]
df_melted_llama2_70 = df_melted[df_melted$'model' == 'llama-2-70b-chat',]
df_melted_mistral7b = df_melted[df_melted$'model' == 'mistral-7b-instruct',]
df_melted_mixtral8x7b = df_melted[df_melted$'model' == 'mixtral-8x7b-instruct',]

####### LLM Model geminipro

logit_model_df_melted_geminipro <- glm(correct ~ spacing + prompt + item, 
                                       family = binomial, data = df_melted_geminipro)
logit_model_df_melted_geminipro_reduced <- glm(correct ~ prompt + item, 
                                       family = binomial, data = df_melted_geminipro)
df_melted_geminipro_deviance_diff <- logit_model_df_melted_geminipro_reduced$deviance - logit_model_df_melted_geminipro$deviance
df_melted_geminipro_chi_squared_test <- 1 - pchisq(df_melted_geminipro_deviance_diff, df = 2)

# Print the p-value
print(df_melted_geminipro_chi_squared_test)


# Summary of the logistic regression model
logreg_summary_df_melted_geminipro <-summary(logit_model_df_melted_geminipro)
logreg_summary_df_melted_geminipro



####### LLM Model gpt35_0613

logit_model_df_melted_gpt35_0613 <- glm(correct ~ spacing + prompt + item, 
                                        family = binomial, data = df_melted_gpt35_0613)
logit_model_df_melted_gpt35_0613_reduced <- glm(correct ~ prompt + item, 
                                                family = binomial, data = df_melted_gpt35_0613)
df_melted_gpt35_0613_deviance_diff <- logit_model_df_melted_gpt35_0613_reduced$deviance - logit_model_df_melted_gpt35_0613$deviance
df_melted_gpt35_0613_chi_squared_test <- 1 - pchisq(df_melted_gpt35_0613_deviance_diff, df = 2)

# Print the p-value
print(df_melted_gpt35_0613_chi_squared_test)

# Summary of the logistic regression model
logreg_summary_df_melted_gpt35_0613 <-summary(logit_model_df_melted_gpt35_0613)
logreg_summary_df_melted_gpt35_0613


####### LLM Model gpt35_1106

logit_model_df_melted_gpt35_1106 <- glm(correct ~ spacing + prompt + item, 
                                        family = binomial, data = df_melted_gpt35_1106)
logit_model_df_melted_gpt35_1106_reduced <- glm(correct ~ prompt + item, 
                                                family = binomial, data = df_melted_gpt35_1106)
df_melted_gpt35_1106_deviance_diff <- logit_model_df_melted_gpt35_1106_reduced$deviance - logit_model_df_melted_gpt35_1106$deviance
df_melted_gpt35_1106_chi_squared_test <- 1 - pchisq(df_melted_gpt35_1106_deviance_diff, df = 2)

# Print the p-value
print(df_melted_gpt35_1106_chi_squared_test)

# Summary of the logistic regression model
logreg_summary_df_melted_gpt35_1106 <-summary(logit_model_df_melted_gpt35_1106)
logreg_summary_df_melted_gpt35_1106


####### LLM Model gpt40_0613

logit_model_df_melted_gpt40_0613 <- glm(correct ~ spacing + prompt + item, 
                                        family = binomial, data = df_melted_gpt40_0613)
logit_model_df_melted_gpt40_0613_reduced <- glm(correct ~ prompt + item, 
                                                family = binomial, data = df_melted_gpt40_0613)
df_melted_gpt40_0613_deviance_diff <- logit_model_df_melted_gpt40_0613_reduced$deviance - logit_model_df_melted_gpt40_0613$deviance
df_melted_gpt40_0613_chi_squared_test <- 1 - pchisq(df_melted_gpt40_0613_deviance_diff, df = 2)

# Print the p-value
print(df_melted_gpt40_0613_chi_squared_test)

# Summary of the logistic regression model
logreg_summary_df_melted_gpt40_0613 <-summary(logit_model_df_melted_gpt40_0613)
logreg_summary_df_melted_gpt40_0613


####### LLM Model gpt40_1106

logit_model_df_melted_gpt40_1106 <- glm(correct ~ spacing + prompt + item, 
                                        family = binomial, data = df_melted_gpt40_1106)
logit_model_df_melted_gpt40_1106_reduced <- glm(correct ~ prompt + item, 
                                                family = binomial, data = df_melted_gpt40_1106)
df_melted_gpt40_1106_deviance_diff <- logit_model_df_melted_gpt40_1106_reduced$deviance - logit_model_df_melted_gpt40_1106$deviance
df_melted_gpt40_1106_chi_squared_test <- 1 - pchisq(df_melted_gpt40_1106_deviance_diff, df = 2)

# Print the p-value
print(df_melted_gpt40_1106_chi_squared_test)

# Summary of the logistic regression model
logreg_summary_df_melted_gpt40_1106 <-summary(logit_model_df_melted_gpt40_1106)
logreg_summary_df_melted_gpt40_1106



####### LLM Model llama2_70

logit_model_df_melted_llama2_70 <- glm(correct ~ spacing + prompt + item, 
                                       family = binomial, data = df_melted_llama2_70)
logit_model_df_melted_llama2_70_reduced <- glm(correct ~ prompt + item, 
                                               family = binomial, data = df_melted_llama2_70)
df_melted_llama2_70_deviance_diff <- logit_model_df_melted_llama2_70_reduced$deviance - logit_model_df_melted_llama2_70$deviance
df_melted_llama2_70_chi_squared_test <- 1 - pchisq(df_melted_llama2_70_deviance_diff, df = 2)

# Print the p-value
print(df_melted_llama2_70_chi_squared_test)

# Summary of the logistic regression model
logreg_summary_df_melted_llama2_70 <-summary(logit_model_df_melted_llama2_70)
logreg_summary_df_melted_llama2_70


####### LLM Model mistral7b

logit_model_df_melted_mistral7b <- glm(correct ~ spacing + prompt + item, 
                                       family = binomial, data = df_melted_mistral7b)
logit_model_df_melted_mistral7b_reduced <- glm(correct ~ prompt + item, 
                                               family = binomial, data = df_melted_mistral7b)
df_melted_mistral7b_deviance_diff <- logit_model_df_melted_mistral7b_reduced$deviance - logit_model_df_melted_mistral7b$deviance
df_melted_mistral7b_chi_squared_test <- 1 - pchisq(df_melted_mistral7b_deviance_diff, df = 2)

# Print the p-value
print(df_melted_mistral7b_chi_squared_test)

# Summary of the logistic regression model
logreg_summary_df_melted_mistral7b <-summary(logit_model_df_melted_mistral7b)
logreg_summary_df_melted_mistral7b


####### LLM Model mixtral8x7b

logit_model_df_melted_mixtral8x7b <- glm(correct ~ spacing + prompt + item, 
                                         family = binomial, data = df_melted_mixtral8x7b)
logit_model_df_melted_mixtral8x7b_reduced <- glm(correct ~ prompt + item, 
                                                 family = binomial, data = df_melted_mixtral8x7b)
df_melted_mixtral8x7b_deviance_diff <- logit_model_df_melted_mixtral8x7b_reduced$deviance - logit_model_df_melted_mixtral8x7b$deviance
df_melted_mixtral8x7b_chi_squared_test <- 1 - pchisq(df_melted_mixtral8x7b_deviance_diff, df = 2)

# Print the p-value
print(df_melted_mixtral8x7b_chi_squared_test)

# Summary of the logistic regression model
logreg_summary_df_melted_mixtral8x7b <-summary(logit_model_df_melted_mixtral8x7b)
logreg_summary_df_melted_mixtral8x7b



