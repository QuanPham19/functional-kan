# Yield curve forecasting - Literature review

## 1. List of references:

[1] A comparison of multitask and single task learning with artificial neural networks for yield curve (Nunes, 2018)

[2] Forecasting the term structure of government bond yields (Diebold and Li, 2006)

[3] Yield curve prediction for the strategic investor (Bernadell, 2005)

[4] A Quantitative Comparison of Yield Curve Models in the MINT Economies (Ayliffe, 2020)

## 2. Construction of yield curves:

Let $P_t(\tau)$ be price of $\tau$-period bond, and $y_t(\tau)$ be zero-coupon nominal yield to maturity. We obtain 2 objectives:
- Forward rate curve: $f_t(\tau) = -P'_t(\tau) / P_t(\tau)$
- Yield to maturity: $y_t(\tau) = \frac{1}{\tau} \int_0^\tau f_t(u) \, du,$

## 3. Modeling approaches:

In terms of methodology, 3 ideas are often considered: Statistical Modeling, Deep Learning and Functional Analysis. There are limited reliable resources on fixed income market:
- Extended Nelson-Siegel curve [2] considers yield curve as 3-factor model of level, slope and curvature
- Multi-layer perceptron network [1] incorporating massive macroeconomics data

Regarding forecasting procedure, 2 approaches are considerable:
- Direct: train 1 model for each forward horizon
- Iterative: train only 1 model and output of last prediction becomes feature of next prediction
- Concern: direct is better to eliminate recursive noises, but iterative is better to reduce runtime