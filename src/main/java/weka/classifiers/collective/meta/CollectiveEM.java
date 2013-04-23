/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * CollectiveEM.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.meta;

import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.collective.CollectiveRandomizableSingleClassifierEnhancer;
import weka.classifiers.collective.util.CollectiveHelper;
import weka.classifiers.evaluation.CollectiveEvaluationUtils;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 <!-- globalinfo-start -->
 * The given base classifier trains on the training set and then labels the instances of the test set. I.e., the original test set is duplicated and each part is assigned one class, the distribution of the classifier is used to set the weight of the instances in each part.<br/>
 * The weights from then on are calculated with this formula:<br/>
 *   mean(n+1) = q * mean(n) + (1-q) * dist(n)<br/>
 * (where q is provided by the user)<br/>
 * Due to keeping track of the mean and using that in the calculation of the new weight, one avoids oscillating behavior.
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -I &lt;num&gt;
 *  Number of iterations.
 *  (default 10)</pre>
 * 
 * <pre> -U
 *  Turns on the updating of the labels/weights of instances
 *  of the training set.
 *  (default off)</pre>
 * 
 * <pre> -Q
 *  The factor for averaging the distribution made by the
 *  classifier.
 *  (default 1)</pre>
 * 
 * <pre> -R
 *  Turns on the randomization of the data before it is
 *  presented to the classifier.
 *  (default off)</pre>
 * 
 * <pre> -output-eval
 *  Appends the evaluations for each iteration to the file
 *  CollectiveEM-Evaluation.
 *  (default off)</pre>
 * 
 * <pre> -folds &lt;folds&gt;
 *  The number of folds for splitting the training set into
 *  train and test set. The first fold is always the training
 *  set. With '-V' you can invert this, i.e., instead of 20/80
 *  for 5 folds you'll get 80/20.
 *  (default 5)</pre>
 * 
 * <pre> -V
 *  Inverts the fold selection, i.e., instead of using the first
 *  fold for the training set it is used for test set and the
 *  remaining folds for training.</pre>
 * 
 * <pre> -verbose
 *  Whether to print some more information during building the
 *  classifier.
 *  (default is off)</pre>
 * 
 * <pre> -verbose
 *  Whether to print some more information during building the
 *  classifier.
 *  (default is off)</pre>
 * 
 * <pre> -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)</pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 * <pre> -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.trees.J48)</pre>
 * 
 * <pre> 
 * Options specific to classifier weka.classifiers.trees.J48:
 * </pre>
 * 
 * <pre> -U
 *  Use unpruned tree.</pre>
 * 
 * <pre> -C &lt;pruning confidence&gt;
 *  Set confidence threshold for pruning.
 *  (default 0.25)</pre>
 * 
 * <pre> -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf.
 *  (default 2)</pre>
 * 
 * <pre> -R
 *  Use reduced error pruning.</pre>
 * 
 * <pre> -N &lt;number of folds&gt;
 *  Set number of folds for reduced error
 *  pruning. One fold is used as pruning set.
 *  (default 3)</pre>
 * 
 * <pre> -B
 *  Use binary splits only.</pre>
 * 
 * <pre> -S
 *  Don't perform subtree raising.</pre>
 * 
 * <pre> -L
 *  Do not clean up after the tree has been built.</pre>
 * 
 * <pre> -A
 *  Laplace smoothing for predicted probabilities.</pre>
 * 
 * <pre> -Q &lt;seed&gt;
 *  Seed for random data shuffling (default 1).</pre>
 * 
 <!-- options-end -->
 *
 * Options after -- are passed to the designated classifier.<p/>
 *
 * @author Bernhard Pfahringer (bernhard at cs dot waikato dot ac dot nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $ 
 */
public class CollectiveEM
  extends CollectiveRandomizableSingleClassifierEnhancer {
  
  /** for serialization */
  private static final long serialVersionUID = 2167889698898023051L;

  /** The number of iterations. */
  protected int m_NumIterations;

  /** The current iteration */
  protected int m_CurrentIteration;

  /** The new training set (weights and labels for each class label) */
  protected Instances[] m_TrainsetNew;

  /** The (current) distribution for each instance in the training set. */
  protected double[][] m_TrainDistCurrent;

  /** The (sum) distribution for each instance in the training set. */
  protected double[][] m_TrainDistSum;

  /** The new test set (weights and labels for each class label) */
  protected Instances[] m_TestsetNew;

  /** The (current) distribution for each instance in the test set. */
  protected double[][] m_TestDistCurrent;

  /** The (sum) distribution for each instance in the test set. */
  protected double[][] m_TestDistSum;

  /** The actual training set to learn from */
  protected Instances m_TrainsetPool;

  /** The evaluation of the training set. */
  protected double m_TrainEval;

  /** The evaluation of the test set. */
  protected double m_TestEval;

  /** The iteration in which the last change happened (regarding the evaluation
   * of the distributions) */
  protected int m_LastChange;

  /** whether to update the labels/weights on the training set or not */
  protected boolean m_UpdateTrainingSet;
  
  /** the value used for averaging the distributions */
  protected double m_QValue;
  
  /** whether to output evaluations in each iteration into a file
   * @see #m_OutputFilename */
  protected boolean m_OutputEvaluations;
  
  /** The output file where to store the evaluation values. Extensions for
   * train and test are added automatically */
  protected String m_OutputFilename;
  
  /** stores the evaluation values of each iteration */
  protected String m_Evaluation[];

  /** whether to randomize the data before presenting it to the classifier */
  protected boolean m_Randomize;

  /** for randomizing the data */
  protected Random m_Random;

  /** the number of classes */
  protected int m_NumClasses;

  /**
   * performs initialization of members
   */
  @Override
  protected void initializeMembers() {
    super.initializeMembers();

    m_Classifier        = new weka.classifiers.trees.J48();
    m_NumIterations     = 10;
    m_CurrentIteration  = -1;
    m_TrainsetNew       = null;
    m_TrainDistCurrent  = null;
    m_TrainDistSum      = null;
    m_TestsetNew        = null;
    m_TestDistCurrent   = null;
    m_TestDistSum       = null;
    m_TrainsetPool      = null;
    m_TrainEval         = 0.0;
    m_TestEval          = 0.0;
    m_LastChange        = -1;
    m_UpdateTrainingSet = false;
    m_QValue            = 1.0;
    m_OutputEvaluations = false;
    m_OutputFilename    = "CollectiveEM-Evaluation";
    m_Evaluation        = new String[]{"", ""};
    m_Randomize         = false;

    m_AdditionalMeasures.add("measureLastChange");
    m_AdditionalMeasures.add("measureTrainEval");
    m_AdditionalMeasures.add("measureTestEval");
  }

  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return   "The given base classifier trains on the training set and then "
           + "labels the instances of the test set. I.e., the original test "
           + "set is duplicated and each part is assigned one class, the "
           + "distribution of the classifier is used to set the weight of the "
           + "instances in each part.\n"
           + "The weights from then on are calculated with this formula:\n"
           + "  mean(n+1) = q * mean(n) + (1-q) * dist(n)\n"
           + "(where q is provided by the user)\n"
           + "Due to keeping track of the mean and using that in the "
           + "calculation of the new weight, one avoids oscillating behavior.";
  }
  
  /**
   * String describing default classifier.
   * 
   * @return		the classname
   */
  @Override
  protected String defaultClassifierString() {
    return weka.classifiers.trees.J48.class.getName();
  }
  
  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration listOptions() {
    Vector result = new Vector();
    
    result.addElement(new Option(
          "\tNumber of iterations.\n"
        + "\t(default 10)",
        "I", 1, "-I <num>"));
    
    result.addElement(new Option(
          "\tTurns on the updating of the labels/weights of instances\n"
        + "\tof the training set.\n"
        + "\t(default off)",
        "U", 0, "-U"));
    
    result.addElement(new Option(
          "\tThe factor for averaging the distribution made by the\n"
        + "\tclassifier.\n"
        + "\t(default 1)",
        "Q", 1, "-Q"));

    result.addElement(new Option(
          "\tTurns on the randomization of the data before it is\n"
        + "\tpresented to the classifier.\n"
        + "\t(default off)",
        "R", 0, "-R"));

    result.addElement(new Option(
          "\tAppends the evaluations for each iteration to the file\n"
        + "\t" + m_OutputFilename + ".\n"
        + "\t(default off)",
        "output-eval", 0, "-output-eval"));
    
    Enumeration en = super.listOptions();
    while (en.hasMoreElements())
      result.addElement(en.nextElement());

    return result.elements();
  }
  
  /**
   * Parses a given list of options. <p/>
   * 
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -I &lt;num&gt;
   *  Number of iterations.
   *  (default 10)</pre>
   * 
   * <pre> -U
   *  Turns on the updating of the labels/weights of instances
   *  of the training set.
   *  (default off)</pre>
   * 
   * <pre> -Q
   *  The factor for averaging the distribution made by the
   *  classifier.
   *  (default 1)</pre>
   * 
   * <pre> -R
   *  Turns on the randomization of the data before it is
   *  presented to the classifier.
   *  (default off)</pre>
   * 
   * <pre> -output-eval
   *  Appends the evaluations for each iteration to the file
   *  CollectiveEM-Evaluation.
   *  (default off)</pre>
   * 
   * <pre> -folds &lt;folds&gt;
   *  The number of folds for splitting the training set into
   *  train and test set. The first fold is always the training
   *  set. With '-V' you can invert this, i.e., instead of 20/80
   *  for 5 folds you'll get 80/20.
   *  (default 5)</pre>
   * 
   * <pre> -V
   *  Inverts the fold selection, i.e., instead of using the first
   *  fold for the training set it is used for test set and the
   *  remaining folds for training.</pre>
   * 
   * <pre> -verbose
   *  Whether to print some more information during building the
   *  classifier.
   *  (default is off)</pre>
   * 
   * <pre> -verbose
   *  Whether to print some more information during building the
   *  classifier.
   *  (default is off)</pre>
   * 
   * <pre> -S &lt;num&gt;
   *  Random number seed.
   *  (default 1)</pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   * <pre> -W
   *  Full name of base classifier.
   *  (default: weka.classifiers.trees.J48)</pre>
   * 
   * <pre> 
   * Options specific to classifier weka.classifiers.trees.J48:
   * </pre>
   * 
   * <pre> -U
   *  Use unpruned tree.</pre>
   * 
   * <pre> -C &lt;pruning confidence&gt;
   *  Set confidence threshold for pruning.
   *  (default 0.25)</pre>
   * 
   * <pre> -M &lt;minimum number of instances&gt;
   *  Set minimum number of instances per leaf.
   *  (default 2)</pre>
   * 
   * <pre> -R
   *  Use reduced error pruning.</pre>
   * 
   * <pre> -N &lt;number of folds&gt;
   *  Set number of folds for reduced error
   *  pruning. One fold is used as pruning set.
   *  (default 3)</pre>
   * 
   * <pre> -B
   *  Use binary splits only.</pre>
   * 
   * <pre> -S
   *  Don't perform subtree raising.</pre>
   * 
   * <pre> -L
   *  Do not clean up after the tree has been built.</pre>
   * 
   * <pre> -A
   *  Laplace smoothing for predicted probabilities.</pre>
   * 
   * <pre> -Q &lt;seed&gt;
   *  Seed for random data shuffling (default 1).</pre>
   * 
   <!-- options-end -->
   *
   * Options after -- are passed to the designated classifier.<p/>
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {
    String tmpStr;
    
    tmpStr = Utils.getOption('I', options);
    if (tmpStr.length() != 0)
      setNumIterations(Integer.parseInt(tmpStr));
    else
      setNumIterations(10);
    
    setUpdateTrainingSet(Utils.getFlag('U', options));
    
    tmpStr = Utils.getOption('Q', options);
    if (tmpStr.length() != 0)
      setQValue(Double.parseDouble(tmpStr));
    else
      setQValue(1);
    
    setOutputEvaluations(Utils.getFlag("output-eval", options));
    
    setRandomize(Utils.getFlag('R', options));
    
    super.setOptions(options);
  }
  
  /**
   * Gets the current settings of the classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String[] getOptions() {
    Vector        result;
    String[]      options;
    int           i;
    
    result  = new Vector();

    result.add("-I");
    result.add("" + getNumIterations());

    if (getUpdateTrainingSet())
      result.add("-U");
    
    if (getOutputEvaluations())
      result.add("-output-eval");
    
    result.add("-Q"); 
    result.add("" + getQValue());

    if (getRandomize())
      result.add("-R");
    
    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);
    
    return (String[]) result.toArray(new String[result.size()]);
  }

  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = new Capabilities(this);

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.DATE_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.BINARY_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);
    
    return result;
  }

  /**
   * Returns the value of the named measure
   * @param measureName the name of the measure to query for its value
   * @return the value of the named measure
   * @throws IllegalArgumentException if the named measure is not supported
   */
  @Override
  public double getMeasure(String measureName) {
    if (measureName.equalsIgnoreCase("measureLastChange"))
      return m_LastChange;
    else if (measureName.equalsIgnoreCase("measureTrainEval"))
      return m_TrainEval;
    else if (measureName.equalsIgnoreCase("measureTestEval"))
      return m_TestEval;
    else
      return super.getMeasure(measureName);
  }
  
  /**
   * Sets the number of iterations
   * 
   * @param numIterations	the number of iterations
   */
  public void setNumIterations(int numIterations) {
    if (numIterations >= 1)
      m_NumIterations = numIterations;
    else
      System.out.println("Must have at least 1 iteration (provided: " +
        numIterations + ")!");
  }
  
  /**
   * Gets the number of iterations
   *
   * @return the number of iterations
   */
  public int getNumIterations() {
    return m_NumIterations;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String numIterationsTipText() {
    return "The number of iterations to be performed.";
  }
  
  /**
   * Sets whether the instances of the training set are updated
   * 
   * @param update	whether instances are updated
   */
  public void setUpdateTrainingSet(boolean update) {
    m_UpdateTrainingSet = update;
  }
  
  /**
   * Gets whether the training set is updated or not
   *
   * @return whether the training instances are updated or not
   */
  public boolean getUpdateTrainingSet() {
    return m_UpdateTrainingSet;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String updateTrainingSetTipText() {
    return "Turns on the updating of weights/labels of the training set.";
  }
  
  /**
   * Sets whether the evaluations are stored in a file
   * 
   * @param output	if true the evaluations are stored in a file
   * @see 		#m_OutputFilename
   */
  public void setOutputEvaluations(boolean output) {
    m_OutputEvaluations = output;
  }
  
  /**
   * Gets whether the evaluations are stored in a file or not
   *
   * @return whether the evaluations are stored in a file
   * @see #m_OutputFilename
   */
  public boolean getOutputEvaluations() {
    return m_OutputEvaluations;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String outputEvaluationsTipText() {
    return "Turns on the storing of the evaluations in a temp. file (" + m_OutputFilename + ").";
  }
  
  /**
   * Sets the value for averaging the distributions
   * 
   * @param value	the value for averaging the distributions
   */
  public void setQValue(double value) {
    if ( (value >= 0) && (value <= 1) )
      m_QValue = value;
    else
      System.out.println("0 <= Q <= 1 (provided: " + value + ")!");
  }
  
  /**
   * Gets the value for averaging the distributions
   *
   * @return the value for averaging the distributions
   */
  public double getQValue() {
    return m_QValue;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String QValueTipText() {
    return "The value used for averaging the distributions of the classifier.";
  }
  
  /**
   * Sets whether the data is randomized before it is presented to the
   * classifier
   * 
   * @param randomize		if true the data is randomized initially
   */
  public void setRandomize(boolean randomize) {
    m_Randomize = randomize;
  }
  
  /**
   * Gets whether the data is randomize before it is presented to the
   * classifier
   *
   * @return whether the training data is randomized
   */
  public boolean getRandomize() {
    return m_Randomize;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String randomizeTipText() {
    return "Turns on the randomizing of the data before presenting it to the classifier.";
  }

  /**
   * resets the classifier
   */
  @Override
  public void reset() {
    super.reset();

    m_CurrentIteration = 1;
    m_LastChange       = 1;
    m_TrainsetNew      = null;
    m_TestsetNew       = null;
  }
  
  /**
   * internal function for determining the class distribution for an instance, 
   * will be overridden by derived classes. For more details about the returned 
   * array, see <code>Classifier.distributionForInstance(Instance)</code>.
   * 
   * @param instance	the instance to get the distribution for
   * @return		the distribution for the given instance
   * @see 		weka.classifiers.Classifier#distributionForInstance(Instance)
   * @throws Exception	if something goes wrong
   */
  @Override
  protected double[] getDistribution(Instance instance) throws Exception {
    return m_Classifier.distributionForInstance(instance);  
  }
  
  /**
   * builds the new train/test instances
   * @throws Exception if anything goes wrong
   */
  @Override
  protected void generateSets() throws Exception {
    int       i;
    int       n;

    if (m_TrainsetNew == null) {
      m_NumClasses  = m_Trainset.numClasses();
      m_TrainsetNew = new Instances[m_NumClasses];
      m_TestsetNew  = new Instances[m_NumClasses];
    }

    super.generateSets();

    if (m_CurrentIteration == 1) {
      m_TrainsetPool   = new Instances(m_Trainset);
      for (i = 0; i < m_NumClasses; i++) {
        m_TrainsetNew[i] = new Instances(m_Trainset);
        m_TestsetNew[i]  = new Instances(m_Testset);
      }
      m_Random         = m_TrainsetPool.getRandomNumberGenerator(getSeed());
    }
    else {
      if (getUpdateTrainingSet()) {
        m_TrainsetPool   = new Instances(m_Trainset, 0);
        for (n = 0; n < m_NumClasses; n++) {
          for (i = 0; i < m_TrainsetNew[n].numInstances(); i++)
            m_TrainsetPool.add(m_TrainsetNew[n].instance(i));
        }
      }
      else {
        m_TrainsetPool   = new Instances(m_Trainset);
        for (i = 0; i < m_NumClasses; i++)
          m_TrainsetNew[i] = new Instances(m_Trainset);
      }
      // add newly labeled instances from both test sets
      for (n = 0; n < m_NumClasses; n++) {
        for (i = 0; i < m_TestsetNew[n].numInstances(); i++)
          m_TrainsetPool.add(m_TestsetNew[n].instance(i));
      }
    }

    // randomize?
    if (getRandomize())
      m_TrainsetPool.randomize(m_Random);
    
    m_TrainDistCurrent = new double[m_Trainset.numInstances()][m_NumClasses];
    m_TestDistCurrent  = new double[m_Testset.numInstances()][m_NumClasses];
  }
  
  /**
   * performs the actual building of the classifier
   * @throws Exception if building fails
   */
  @Override
  protected void buildClassifier() throws Exception {
    m_Classifier.buildClassifier(m_TrainsetPool);
  }

  /**
   * determines the distributions for the given instances
   * 
   * @param instances	the instances to determine the distribution for
   * @param dist	the disttribution to fill
   * @throws Exception	if something goes wrong
   */
  protected void determineDistributions(Instances instances, double[][] dist) 
    throws Exception {

    int         i;
    int         n;
    double[]    d;

    for (i = 0; i < dist.length; i++) {
      d = distributionForInstance(instances.instance(i));
      for (n = 0; n < d.length; n++)
        dist[i][n] = d[n];
    }
  }

  /**
   * sets new labels and weights for the given instances based on the
   * distributions
   * @param sets            the sets to work on
   * @param distsum         the summed up distributions
   * @param dist            the distributions (generated by a classifier)
   */
  protected void setLabelsAndWeights( Instances[] sets, 
                                      double[][] distsum, 
                                      double[][] dist ) {
    int         i;
    int         n;
    String[]    clsValues;
    double[]    means;
    double[]    weights;
    double      q;

    // get class labels
    clsValues = new String[m_NumClasses];
    n         = sets[0].classIndex();
    for (i = 0; i < m_NumClasses; i++)
      clsValues[i] = sets[0].classAttribute().value(i);
    
    // set labels/weights
    means   = new double[m_NumClasses];
    weights = new double[m_NumClasses];
    q       = getQValue();
    for (i = 0; i < sets[0].numInstances(); i++) {
      // labels
      if (m_CurrentIteration == 1) {
        for (n = 0; n < m_NumClasses; n++)
          sets[n].instance(i).setClassValue(clsValues[n]);
      }

      // weights
      for (n = 0; n < m_NumClasses; n++) {
        means[n]   = distsum[i][n];
        weights[n] = dist[i][n];
      }
      Utils.normalize(means);
      
      if (Utils.eq(q, 1.0)) {
        for (n = 0; n < m_NumClasses; n++)
          sets[n].instance(i).setWeight(means[n]);
      }
      else {
        for (n = 0; n < m_NumClasses; n++)
          sets[n].instance(i).setWeight(q * means[n] + (1-q) * weights[n]);
      }
    }
  }

  /**
   * calculates sum((1 - max-p)^2), where max-p is the highest of the
   * probabilities and sum means the sum over all distributions.
   * 
   * @param dist	the distribution to base the evaluation on
   * @return		the result of the calculation
   */
  protected double evaluateDistributions(double[][] dist) {
    double        result;
    int           i;

    result = 0;

    for (i = 0; i < dist.length; i++)
      result += StrictMath.pow(dist[i][Utils.maxIndex(dist[i])], 2);

    return result;
  }
  
  /**
   * performs the iterations in order to improve the classifier
   * 
   * @throws Exception	if something goes wrong
   */
  @Override
  protected void build() throws Exception {
    double        oldTrainEval;
    double        oldTestEval;
    int           i;
    int           n;

    oldTrainEval   = 0;
    oldTestEval    = 0;
    m_TrainDistSum = new double[m_Trainset.numInstances()][m_NumClasses];
    m_TestDistSum  = new double[m_Testset.numInstances()][m_NumClasses];

    while (m_CurrentIteration <= m_NumIterations) {
      if (getVerbose())
        System.out.println("Iteration " + m_CurrentIteration + "/" 
                           + m_NumIterations + ":");
      
      // build datasets
      generateSets();
       
      // build classifier
      buildClassifier();

      // determine distributions
      determineDistributions(m_Trainset, m_TrainDistCurrent);
      for (n = 0; n < m_NumClasses; n++) {
        for (i = 0; i < m_TrainDistCurrent.length; i++)
          m_TrainDistSum[i][n] += m_TrainDistCurrent[i][n];
      }
      determineDistributions(m_Testset,  m_TestDistCurrent);
      for (n = 0; n < m_NumClasses; n++) {
        for (i = 0; i < m_TestDistCurrent.length; i++)
          m_TestDistSum[i][n] += m_TestDistCurrent[i][n];
      }

      // evaluate the distribution
      m_TrainEval = evaluateDistributions(m_TrainDistCurrent);
      m_TestEval  = evaluateDistributions(m_TestDistCurrent);
      if (getVerbose()) {
        System.out.print(" - train-eval: " + Utils.doubleToString(m_TrainEval, 4));
        if (m_CurrentIteration > 1)
          System.out.println(" (" + Utils.doubleToString(m_TrainEval - oldTrainEval, 4) + ")");
        else
          System.out.println();
        System.out.print(" - test-eval : " + Utils.doubleToString(m_TestEval, 4));
        if (m_CurrentIteration > 1)
          System.out.println(" (" + Utils.doubleToString(m_TestEval  - oldTestEval, 4)  + ")");
        else
          System.out.println();
      }
      if ( (oldTrainEval != m_TrainEval) || (oldTestEval != m_TestEval) )
        m_LastChange = m_CurrentIteration;
      oldTrainEval = m_TrainEval;
      oldTestEval  = m_TestEval;
      
      // set labels/weights
      setLabelsAndWeights(m_TrainsetNew, m_TrainDistSum, m_TrainDistCurrent);
      setLabelsAndWeights(m_TestsetNew,  m_TestDistSum,  m_TestDistCurrent);
      
      // store evaluations in file?
      if ( (m_CurrentIteration > 1) && (getOutputEvaluations()) ) {
        if (!m_Evaluation[0].equals("")) {
          m_Evaluation[0] += " ";
          m_Evaluation[1] += " ";
        }
        m_Evaluation[0] += Utils.doubleToString(m_TrainEval, 4);
        m_Evaluation[1] += Utils.doubleToString(m_TestEval,  4);
      }
      
      // next iteration
      m_CurrentIteration++;
    }
    
    // write output of evaluations to file
    if (getOutputEvaluations()) {
      String relName = m_Trainset.relationName();
      relName = relName.replaceAll("-weka\\.filters\\..*", "");
      CollectiveHelper.writeToTempFile(
          m_OutputFilename + "-" + relName + "-" + getQValue() + ".train", 
          m_Evaluation[0], true);
      CollectiveHelper.writeToTempFile(
          m_OutputFilename + "-" + relName + "-" + getQValue() + ".test",  
          m_Evaluation[1], true);
      m_Evaluation[0] = "";
      m_Evaluation[1] = "";
    }
  }
  
  /**
   * returns information about the classifier(s)
   * 
   * @return		information about the classifier
   */
  @Override
  protected String toStringClassifier() {
    String        result;

    result  = super.toStringClassifier();
    result += "Iter. with last change: " + m_LastChange + "\n";

    return result;
  }
  
  /**
   * returns the best model as string representation. derived classes have to 
   * add additional information here, like printing the classifier etc.
   * 
   * @return		the string representation of the best model
   */
  @Override
  protected String toStringModel() {
    StringBuffer     text;
    
    text = new StringBuffer();
    text.append(super.toStringModel());
    text.append("\n");
    text.append(getClassifier().toString());
    
    return text.toString();
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 2019 $");
  }
  
  /**
   * Main method for testing this class.
   *
   * @param args the options
   */
  public static void main(String[] args) {
    CollectiveEvaluationUtils.runClassifier(new CollectiveEM(), args);
  }
}
