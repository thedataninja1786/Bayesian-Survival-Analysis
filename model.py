class BS():
  def __init__(self,df,event,time):
    self.df = df
    self.total_steps = sorted(df[time].unique())
    self.event = event 
    self.time = time 
    self.survival_probabilities = [] 

  def baseline_hazard(self):
    posteriors = [[1 for x in range(0,self.df.shape[1]-2)]]
    print(f'There are {len(posteriors[0])} covariates present.')
    events = self.df[self.df[self.event] == 1]
    event_steps = sorted(events[self.time].unique())

    for t in self.total_steps:
      if t not in event_steps:
        posteriors.append(posteriors[-1]) #probs are the same as before 
        continue # in the case the patient left -> continue 

      current_probs = []
      prior_entries = self.df[self.df[self.time] <= t]
      at_risk = self.df[self.df[self.time] == t]
      i = -1
      for col in self.df:
        if len(set(self.df[col].values)) > 0 and col not in [self.time,self.event]:
          prior_entries[col] = pd.to_numeric(prior_entries[col])
          i += 1
          sample_mean = prior_entries[col].mean()
          sample_sd = max(prior_entries[col].std(),1)
          prob = 1
          for x in at_risk[col]:
            prob += abs(np.log((1/(sample_sd * (6.28 ** 0.5)))) - (-0.5 * (((x - sample_mean) / sample_sd) ** 2))) / 1000
          current_probs.append(prob * posteriors[-1][i])
      posteriors.append(current_probs)

    for row in posteriors:
      surv_prob = 0
      for p in row:
        surv_prob +=  np.log(p) 
      self.survival_probabilities.append(abs(surv_prob)) 
    self.survival_probabilities = [1 - (x / max(self.survival_probabilities)) for x in self.survival_probabilities]
    return self.total_steps, self.survival_probabilities[:-1]

  def _estimate_weights(self):
    from sklearn.linear_model import LogisticRegression
    X = self.df[:]
    del X[self.event]; del X[self.time]
    y = df[self.event].values
    clf = LogisticRegression(random_state=0).fit(X, y)
    return clf.coef_

  def individual_surv(self,idx):
    df = self.df[:]
    weights = self._estimate_weights()
    t = df[self.time].to_numpy(); e = df[self.event].to_numpy()
    del df[self.time]; del df[self.event]
    covariates = df.to_numpy()
    covariates = covariates.dot(weights.T)
    surv = covariates.squeeze() * t/(max(t))
    surv = [abs(x/max(surv)) for x in surv]
    df[self.time] = t; df[self.event] = e; df['surv'] = surv
    ind_surv = [min(x + surv[idx],1) for x in self.survival_probabilities[:-1]]
    plt.step(self.total_steps,ind_surv)
    plt.show()
    return ind_surv
  
