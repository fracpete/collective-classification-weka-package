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
 * SimpleCollective.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.meta;

import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.collective.CollectiveRandomizableSingleClassifierEnhancer;
import weka.classifiers.collective.RestartableCollectiveClassifier;
import weka.classifiers.collective.util.CollectiveHelper;
import weka.classifiers.collective.util.CollectiveInstances;
import weka.classifiers.collective.util.CollectiveLog;
import weka.classifiers.collective.util.FlipHistory;
import weka.classifiers.collective.util.Flipper;
import weka.classifiers.collective.util.TriangleFlipper;
import weka.classifiers.evaluation.CollectiveEvaluationUtils;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Utils;

/**
 <!-- globalinfo-start -->
 * Uses collective classification (cf. transduction, semi-supersived learning), i.e., it uses additionally the data given in the test file (with random labels at first) to get the most out of a small training set. It is attempted to improve the initial random labels through several iterations, whereas a restart reinitializes all labels again randomly.
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
 * <pre> -R &lt;num&gt;
 *  Number of restarts.
 *  (default 10)</pre>
 * 
 * <pre> -log
 *  Creates logs in the tmp directory for all kinds of internal data.
 *  Use only for debugging purposes!
 * </pre>
 * 
 * <pre> -U
 *  Updates also the labels of the training set.
 * </pre>
 * 
 * <pre> -eval &lt;num&gt;
 *  The type of evaluation to use (0 = Randomwalk/Last model used for 
 *  prediction, 1=Randomwalk/Best model used for prediction,
 *  2=Hillclimbing).
 * </pre>
 * 
 * <pre> -compare &lt;num&gt;
 *  The type of comparisong used for comparing models.
 *  (0=overall RMS, 1=RMS on train set, 2=RMS on test set, 
 *  3=Accuracy on train set)
 * </pre>
 * 
 * <pre> -flipper "&lt;classname [parameters]&gt;"
 *  The flipping algorithm (and optional parameters) to use for 
 *  flipping labels.
 * </pre>
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
public class SimpleCollective 
  extends CollectiveRandomizableSingleClassifierEnhancer
  implements RestartableCollectiveClassifier {

  /** for serialization */
  private static final long serialVersionUID = 824837327293674316L;

  /** The number of iterations. */
  protected int m_NumIterations;
  
  /** The number of restarts. */
  protected int m_NumRestarts;
  
  /** marks the restart in which the last improvement happened */
  protected int m_LastRestart;
  
  /** marks the iteration in which the last improvement happened */
  protected int m_LastIteration;
  
  /** the current RMS on the train set */
  protected double m_RMSTrain;
  
  /** the current RMS on the test set */
  protected double m_RMSTest;
  
  /** the current RMS on the (original) test set */
  protected double m_RMSTestOriginal;
  
  /** the current RMS on the both sets */
  protected double m_RMS;

  /** the accuracy on the training set */
  protected double m_AccTrain;
  
  /** the accuracy on the (original) test set */
  protected double m_AccTestOriginal;
  
  /** marks the RMS of the last improvement on the train set */
  protected double m_LastRMSTrain;
  
  /** marks the RMS of the last improvement on the test set */
  protected double m_LastRMSTest;
  
  /** marks the RMS of the last improvement on the (original) test set */
  protected double m_LastRMSTestOriginal;
  
  /** marks the RMS of the last improvement on both sets */
  protected double m_LastRMS;
  
  /** marks the accuracy of the last improvement on the train set */
  protected double m_LastAccTrain;
  
  /** used for storing the best model */
  protected SimpleCollective m_BestModel;
  
  /** used for initializing/flipping labels */
  protected CollectiveInstances m_CollectiveInstances;

  /** whether to update the labels of the training set */
  protected boolean m_UpdateTraining;

  /** the type of evaluation to use */
  protected int m_EvaluationType;

  /** the type of comparison to use */
  protected int m_ComparisonType;

  /** the flipping algorithm to use */
  protected Flipper m_Flipper;

  /** whether to log internal stuff */
  protected boolean m_Log;

  /** for storing the log-entries */
  protected CollectiveLog m_LogEntries;

  /** the new training set */
  protected Instances m_TrainsetNew;

  /** the flipping history */
  protected FlipHistory m_FlipHistory;

  /**
   * performs initialization of members
   */
  @Override
  protected void initializeMembers() {
    super.initializeMembers();

    m_NumIterations       = 10;
    m_NumRestarts         = 10;
    m_LastRestart         = -1;
    m_LastIteration       = -1;
    m_RMSTrain            = 1.0;
    m_RMSTest             = 1.0;
    m_RMSTestOriginal     = 1.0;
    m_RMS                 = 1.0;
    m_AccTrain            = 0.0;
    m_AccTestOriginal     = 0.0;
    m_LastRMSTrain        = 1.0;
    m_LastRMSTest         = 1.0;
    m_LastRMSTestOriginal = 1.0;
    m_LastRMS             = 1.0;
    m_LastAccTrain        = 0.0;
    m_BestModel           = null;
    m_CollectiveInstances = null;
    m_UpdateTraining      = false;
    m_EvaluationType      = CollectiveInstances.EVAL_RANDOMWALK_BEST;
    m_ComparisonType      = CollectiveInstances.COMPARE_RMS_TRAIN;
    m_Log                 = false;
    m_TrainsetNew         = null;
    m_FlipHistory         = null;
    m_Classifier          = new weka.classifiers.trees.J48();
    m_Flipper             = new TriangleFlipper();
    m_LogEntries          = new CollectiveLog();
    
    m_AdditionalMeasures.add("measureLastRestart");
    m_AdditionalMeasures.add("measureLastIteration");
    m_AdditionalMeasures.add("measureLastRMSTrain");
    m_AdditionalMeasures.add("measureLastRMSTest");
    m_AdditionalMeasures.add("measureLastRMSTestOriginal");
    m_AdditionalMeasures.add("measureLastRMS");
    m_AdditionalMeasures.add("measureLastAccTrain");
  }

  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return    
      "Uses collective classification (cf. transduction, semi-supersived "
    + "learning), i.e., it uses additionally the data given in the test file "
    + "(with random labels at first) to get the most out of a small training "
    + "set. It is attempted to improve the initial random labels through "
    + "several iterations, whereas a restart reinitializes all labels again "
    + "randomly.";
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
        "\tNumber of restarts.\n"
        + "\t(default 10)",
        "R", 1, "-R <num>"));
    
    result.addElement(new Option(
        "\tCreates logs in the tmp directory for all kinds of internal data.\n"
        + "\tUse only for debugging purposes!\n",
        "log", 0, "-log"));
    
    result.addElement(new Option(
        "\tUpdates also the labels of the training set.\n",
        "U", 0, "-U"));
    
    result.addElement(new Option(
        "\tThe type of evaluation to use (0 = Randomwalk/Last model used for \n"
        + "\tprediction, 1=Randomwalk/Best model used for prediction,\n"
        + "\t2=Hillclimbing).\n",
        "eval", 1, "-eval <num>"));
    
    result.addElement(new Option(
        "\tThe type of comparisong used for comparing models.\n"
        + "\t(0=overall RMS, 1=RMS on train set, 2=RMS on test set, \n"
        + "\t3=Accuracy on train set)\n",
        "compare", 1, "-compare <num>"));
    
    result.addElement(new Option(
        "\tThe flipping algorithm (and optional parameters) to use for \n"
        + "\tflipping labels.\n",
        "flipper", 1, "-flipper \"<classname [parameters]>\""));
    
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
   * <pre> -R &lt;num&gt;
   *  Number of restarts.
   *  (default 10)</pre>
   * 
   * <pre> -log
   *  Creates logs in the tmp directory for all kinds of internal data.
   *  Use only for debugging purposes!
   * </pre>
   * 
   * <pre> -U
   *  Updates also the labels of the training set.
   * </pre>
   * 
   * <pre> -eval &lt;num&gt;
   *  The type of evaluation to use (0 = Randomwalk/Last model used for 
   *  prediction, 1=Randomwalk/Best model used for prediction,
   *  2=Hillclimbing).
   * </pre>
   * 
   * <pre> -compare &lt;num&gt;
   *  The type of comparisong used for comparing models.
   *  (0=overall RMS, 1=RMS on train set, 2=RMS on test set, 
   *  3=Accuracy on train set)
   * </pre>
   * 
   * <pre> -flipper "&lt;classname [parameters]&gt;"
   *  The flipping algorithm (and optional parameters) to use for 
   *  flipping labels.
   * </pre>
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
    String        tmpStr;
    String[]      tmpOptions;

    tmpStr = Utils.getOption('I', options);
    if (tmpStr.length() != 0)
      setNumIterations(Integer.parseInt(tmpStr));
    else
      setNumIterations(10);
    
    tmpStr = Utils.getOption('R', options);
    if (tmpStr.length() != 0)
      setNumRestarts(Integer.parseInt(tmpStr));
    else
      setNumRestarts(10);
    
    setLog(Utils.getFlag("log", options));
    
    setUpdateTraining(Utils.getFlag('U', options));
    
    tmpStr = Utils.getOption("eval", options);
    if (tmpStr.length() != 0)
      setEvaluationType(new SelectedTag(
            Integer.parseInt(tmpStr), 
            CollectiveInstances.EVAL_TAGS));
    else
      setEvaluationType(new SelectedTag(
            CollectiveInstances.EVAL_RANDOMWALK_LAST, 
            CollectiveInstances.EVAL_TAGS));
    
    tmpStr = Utils.getOption("compare", options);
    if (tmpStr.length() != 0)
      setComparisonType(new SelectedTag(
            Integer.parseInt(tmpStr), 
            CollectiveInstances.COMPARE_TAGS));
    else
      setComparisonType(new SelectedTag(
            CollectiveInstances.COMPARE_RMS, 
            CollectiveInstances.COMPARE_TAGS));
    
    tmpStr = Utils.getOption("flipper", options);
    if (tmpStr.length() != 0) {
      tmpOptions    = Utils.splitOptions(tmpStr);
      tmpStr        = tmpOptions[0];
      tmpOptions[0] = "";
      setFlipper(Flipper.forName(tmpStr, tmpOptions));
    }
    else {
      setFlipper(new TriangleFlipper());
    }
    
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
    
    result.add("-R");
    result.add("" + getNumRestarts());
    
    if (getLog())
      result.add("-log");
    
    if (getUpdateTraining())
      result.add("-U");
    
    result.add("-eval");
    result.add("" + m_EvaluationType);
    
    result.add("-compare");
    result.add("" + m_ComparisonType);
    
    result.add("-flipper");
    result.add(Flipper.getSpecification(getFlipper()));
    
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
    if (measureName.equalsIgnoreCase("measureLastRestart"))
      return getLastRestart();
    else if (measureName.equalsIgnoreCase("measureLastIteration"))
      return getLastIteration();
    else if (measureName.equalsIgnoreCase("measureLastRMSTrain"))
      return getLastRMSTrain();
    else if (measureName.equalsIgnoreCase("measureLastRMSTest"))
      return getLastRMSTest();
    else if (measureName.equalsIgnoreCase("measureLastRMSTestOriginal"))
      return getLastRMSTestOriginal();
    else if (measureName.equalsIgnoreCase("measureLastRMS"))
      return getLastRMS();
    else if (measureName.equalsIgnoreCase("measureLastAccTrain"))
      return getLastAccTrain();
    else
      return super.getMeasure(measureName);
  }
  
  /**
   * Sets the number of iterations
   * 
   * @param value	the number of iterations
   */
  public void setNumIterations(int value) {
    if (value >= 1)
      m_NumIterations = value;
    else
      System.out.println("Must have at least 1 iteration (provided: " +
        value + ")!");
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
   * Sets the number of restarts
   * 
   * @param value	the number of restarts
   */
  public void setNumRestarts(int value) {
    if (value >= 1)
      m_NumRestarts = value;
    else
      System.out.println("Must have at least 1 restart (provided: " +
        value + ")!");
  }
  
  /**
   * Gets the number of restarts
   *
   * @return the number of restarts
   */
  public int getNumRestarts() {
    return m_NumRestarts;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String numRestartsTipText() {
    return "The number of restarts to be performed.";
  }
  
  /**
   * Sets whether to use log internal data to tmp directory.
   *
   * @param value whether to log internal stuff
   */
  public void setLog(boolean value) {
    m_Log = value;
  }
  
  /**
   * Gets whether internal data is logged to tmp directory.
   *
   * @return returns TRUE if internal data is logged to tmp directory
   */
  public boolean getLog() {
    return m_Log;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String logTipText() {
    return   "Whether to log internal data to the tmp directory (" 
           + CollectiveHelper.getTempPath() + ").";
  }
  
  /**
   * Sets the updating of the training labels.
   *
   * @param value true if the labels of the training set should be updated, too
   */
  public void setUpdateTraining(boolean value) {
    m_UpdateTraining = value;
  }
  
  /**
   * Gets whether the labels of the training set are updated, too.
   *
   * @return true if labels of the training set are updated, too
   */
  public boolean getUpdateTraining() {
    return m_UpdateTraining;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String updateTrainingTipText() {
    return "Whether to update labels of the training set.";
  }
     
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String evaluationTypeTipText() {
    return "The type of evaluation to use.";
  }
  
  /**
   * Gets how the classifiers are evaluated.
   *
   * @return the evaluation type.
   */
  public SelectedTag getEvaluationType() {
    return new SelectedTag(m_EvaluationType, CollectiveInstances.EVAL_TAGS);
  }
  
  /**
   * Sets how the classifiers are evaluated.
   *
   * @param tag the evaluation type.
   */
  public void setEvaluationType(SelectedTag tag) {
    if (tag.getTags() == CollectiveInstances.EVAL_TAGS)
      m_EvaluationType = tag.getSelectedTag().getID();
  }
     
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String comparisonTypeTipText() {
    return "The type of comparison to use for comparing models.";
  }
  
  /**
   * Gets how the models are compared.
   *
   * @return the comparison type.
   */
  public SelectedTag getComparisonType() {
    return new SelectedTag(m_ComparisonType, CollectiveInstances.COMPARE_TAGS);
  }
  
  /**
   * Sets how the models are compared.
   *
   * @param value the comparison type.
   */
  public void setComparisonType(SelectedTag value) {
    if (value.getTags() == CollectiveInstances.COMPARE_TAGS)
      m_ComparisonType = value.getSelectedTag().getID();
  }
     
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String flipperTipText() {
    return "The flipping algorithm to use for flipping labels.";
  }
  
  /**
   * Gets the flipping algrithm.
   *
   * @return the flipping algorithm.
   */
  public Flipper getFlipper() {
    return m_Flipper;
  }
  
  /**
   * Sets the flipping algorithm.
   *
   * @param value the flipping algorithm type.
   */
  public void setFlipper(Flipper value) {
    m_Flipper = value;
  }

  /**
   * returns best model depending, or if that is null the current classifier
   * 
   * @return		the best model
   */
  protected SimpleCollective getBestModel() {
    if (m_BestModel != null)
      return m_BestModel;
    else
      return this;
  }
  
  /**
   * compares the two classifiers and returns -1 if itself is worse than o, 0 if
   * they both are equally good and 1 if itself ist better than o; if o is not a
   * Collective Classifier, null or an Exception happens it returns 0
   * @param o is the Collective Classifier to compare to
   * @return returns -1 if o is better, 0 if both are equal and 1 if o is worse
   */
  public int compareTo(Object o) {
    int                 result;
    SimpleCollective    c;
    double              valThis;
    double              valOther;
    
    if (    (o == null) 
         || !(o instanceof CollectiveRandomizableSingleClassifierEnhancer) ) {
      result = 0;

      if (getDebug())
        System.out.print("   ??? + ??? -> " + result);
    }
    else {
      c = (SimpleCollective) o;

      // choose comparison type
      switch (m_ComparisonType) {
        case CollectiveInstances.COMPARE_RMS:
          valThis  = m_RMS;
          valOther = c.getLastRMS();
          break;

        case CollectiveInstances.COMPARE_RMS_TRAIN:
          valThis  = m_RMSTrain;
          valOther = c.getLastRMSTrain();
          break;

        case CollectiveInstances.COMPARE_RMS_TEST:
          valThis  = m_RMSTest;
          valOther = c.getLastRMSTest();
          break;

        case CollectiveInstances.COMPARE_ACC_TRAIN:
          valThis  = 1.0 - m_AccTrain;
          valOther = 1.0 - c.getLastAccTrain();
          break;

        default:
          valThis  = m_RMS;
          valOther = c.getLastRMS();
          break;
      }

      // evaluate
      if (valThis < valOther)
        result = 1;
      else if (valThis > valOther)
        result = -1;
      else
        result = 0;

      if (getDebug())
        System.out.print(   "   " + valThis + " + " + valOther
                          + " -> " + result);
    }
    
    return result;
  }
  
  /**
   * resets the classifier, i.e., the best model, last known restart and
   * iteration, etc.
   * 
   * @see #m_BestModel
   * @see #m_LastIteration
   * @see #m_LastRestart
   */
  @Override
  public void reset() {
    super.reset();

    m_BestModel           = null;
    m_LastIteration       = -1;
    m_LastRestart         = -1;
    m_LastRMSTrain        = 1.0;
    m_LastRMSTest         = 1.0;
    m_LastRMSTestOriginal = 1.0;
    m_LastRMS             = 1.0;
    m_RMSTrain            = 1.0;
    m_RMSTest             = 1.0;
    m_RMSTestOriginal     = 1.0;
    m_RMS                 = 1.0;
    m_AccTrain            = 0.0;
    m_AccTestOriginal     = 0.0;
    m_LastAccTrain        = 0.0;
    
    // for logging
    m_LogEntries.clear();
    m_LogEntries.addFilename(
        "RMS",
        createFilename("-rms.csv"));
    m_LogEntries.addFilename(
        "RMSTrain",
        createFilename("-rms_train.csv"));
    m_LogEntries.addFilename(
        "RMSTest",
        createFilename("-rms_test.csv"));
    m_LogEntries.addFilename(
        "RMSTestOriginal",
        createFilename("-rms_test-original.csv"));
    m_LogEntries.addFilename(
        "AccTrain",
        createFilename("-acc_train.csv"));
    m_LogEntries.addFilename(
        "AccTestOriginal",
        createFilename("-acc_test-original.csv"));
    m_LogEntries.addFilename(
        "FlippedLabels",
        createFilename("-flipped.csv"));
  }
  
  /**
   * returns the run in which the last improvement happened
   * 
   * @return		the last restart
   */
  public int getLastRestart() {
    if (m_BestModel != null)
      return getBestModel().getLastRestart();
    else
      return m_LastRestart;
  }  
  
  /**
   * returns the iteration in which the last improvement happened
   * 
   * @return		the last iteration
   */
  public int getLastIteration() {
    if (m_BestModel != null)
      return getBestModel().getLastIteration();
    else
      return m_LastIteration;
  }
  
  /**
   * returns the RMS on the train set in which the last improvement happened
   * 
   * @return		the last RMS on the train set
   */
  public double getLastRMSTrain() {
    if (m_BestModel != null)
      return getBestModel().getLastRMSTrain();
    else
      return m_LastRMSTrain;
  }
  
  /**
   * returns the RMS on the test set in which the last improvement happened
   * 
   * @return		the last RMS on the test set
   */
  public double getLastRMSTest() {
    if (m_BestModel != null)
      return getBestModel().getLastRMSTest();
    else
      return m_LastRMSTest;
  }
  
  /**
   * returns the RMS on the (original) test set in which the last improvement
   * happened
   * 
   * @return		the last RMS on the original test set
   */
  public double getLastRMSTestOriginal() {
    if (m_BestModel != null)
      return getBestModel().getLastRMSTestOriginal();
    else
      return m_LastRMSTestOriginal;
  }
  
  /**
   * returns the RMS on both sets in which the last improvement happened
   * 
   * @return		the last RMS (train + test)
   */
  public double getLastRMS() {
    if (m_BestModel != null)
      return getBestModel().getLastRMS();
    else
      return m_LastRMS;
  }
  
  /**
   * returns the Accuracy on the train set in which the last improvement 
   * happened
   * 
   * @return		the last accuracy on the train set
   */
  public double getLastAccTrain() {
    if (m_BestModel != null)
      return getBestModel().getLastAccTrain();
    else
      return m_LastAccTrain;
  }
  
  /**
   * returns whether the classifier could be improved during the runs/iterations
   * 
   * @return		true if the classifier could be improved
   */
  public boolean classifierImproved() {
    return (m_LastRestart > -1);
  }
  
  /**
   * builds the necessary CollectiveInstances from the given Instances
   * @throws Exception if anything goes wrong
   */
  @Override
  protected void generateSets() throws Exception {
    int       i;
    
    super.generateSets();

    m_CollectiveInstances = new CollectiveInstances();
    m_CollectiveInstances.setSeed(getSeed());
    m_CollectiveInstances.setFlipper(m_Flipper);
    
    m_TrainsetNew = new Instances(m_Trainset);
    for (i = 0; i < m_Testset.numInstances(); i++)
      m_TrainsetNew.add(m_Testset.instance(i));

    m_FlipHistory = new FlipHistory(m_TrainsetNew);
  }
  
  /**
   * initializes the labels of the CollectiveInstances
   * @throws Exception if initialization fails
   */
  public void initializeLabels() throws Exception {
    m_TrainsetNew = m_CollectiveInstances.initializeLabels(
                        m_Trainset, m_TrainsetNew, 
                        m_Trainset.numInstances(), m_Testset.numInstances());
  }
  
  /**
   * flips the labels of the CollectiveInstances
   * @throws Exception if flipping fails
   */
  public void flipLabels() throws Exception {
    int       from;
    int       count;

    if (getUpdateTraining()) {
      from  = 0;
      count = m_TrainsetNew.numInstances();
    }
    else {
      from  = m_Trainset.numInstances();
      count = m_Testset.numInstances();
    }
    
    if (m_EvaluationType == CollectiveInstances.EVAL_HILLCLIMBING)
      m_TrainsetNew = m_CollectiveInstances.flipLabels(
                          getBestModel().getClassifier(), 
                          m_TrainsetNew, from, count,
                          m_FlipHistory );
    else
      m_TrainsetNew = m_CollectiveInstances.flipLabels(
                          this.getClassifier(), 
                          m_TrainsetNew, from, count,
                          m_FlipHistory );
  }
  
  /**
   * internal function for determining the class distribution for an instance, 
   * will be overridden by derived classes. For more details about the returned 
   * array, see <code>Classifier.distributionForInstance(Instance)</code>.
   * 
   * @param instance	the instance to get the distribution for
   * @return		the distribution for the given instance
   * @throws Exception	if something goes wrong
   */
  @Override
  protected double[] getDistribution(Instance instance) throws Exception {
    if (m_EvaluationType == CollectiveInstances.EVAL_RANDOMWALK_LAST) 
      return this.getClassifier().distributionForInstance(instance);  
    else
      return getBestModel().getClassifier().distributionForInstance(instance);  
  }
  
  /**
   * performs the actual building of the classifier
   * @throws Exception if building fails
   */
  @Override
  protected void buildClassifier() throws Exception {
    m_Classifier.buildClassifier(m_TrainsetNew);
  }

  /**
   * calculates the RMS for test and train set
   * 
   * @throws Exception	if something goes wrong
   */
  protected void calculateRMS() throws Exception {
    double[]      rms;
    
    rms = CollectiveInstances.calculateRMS(
              this.getClassifier(), m_Trainset, 
              new Instances(
                  m_TrainsetNew, 
                  m_Trainset.numInstances(), 
                  m_Testset.numInstances()),
              m_TestsetOriginal);

    m_RMS             = rms[0];
    m_RMSTrain        = rms[1];
    m_RMSTest         = rms[2];
    m_RMSTestOriginal = rms[3];
    
    if (getVerbose())
      System.out.println(   "\nRMSTest: "   + m_RMSTest + ", "
                          + "RMSTestOrig: " + m_RMSTestOriginal + ", "
                          + "RMSTrain: "    + m_RMSTrain + ", "
                          + "RMS: "         + m_RMS);
  }

  /**
   * calculates the accuracy for original test and train set
   * 
   * @throws Exception	if something goes wrong
   */
  protected void calculateAccuracy() throws Exception {
    double[]      acc;
    
    acc = CollectiveInstances.calculateAccuracy(
              this.getClassifier(), m_Trainset, m_TestsetOriginal);

    m_AccTrain        = acc[0];
    m_AccTestOriginal = acc[1];
    
    if (getVerbose())
      System.out.println(   "\nAccTrain: "   + m_AccTrain + ", "
                          + "AccTestOrig: " + m_AccTestOriginal);
  }
  
  /**
   * performs the restarts and iterations and therefore the calls to
   * <code>initializeLabels()</code>, <code>flipLabels()</code> and
   * <code>buildClassifier()</code>
   * 
   * @see   #initializeLabels()
   * @see   #flipLabels()
   * @see   #buildClassifier()
   * @throws Exception if something goes wrong
   */
  @Override
  protected void build() throws Exception {
    int          r;
    int          i;
    boolean      firstRun;

    if (getVerbose())
      System.out.println("\nRunning " + m_NumRestarts 
          + " Restarts (= R) with " + m_NumIterations 
          + " Iterations (= I) each.\n");
    
    for (r = 0; r < m_NumRestarts; r++) {
      firstRun = true;
      for (i = 0; i < m_NumIterations; i++) {
        // info
        if (getVerbose())
          System.out.print("Run (R/I) " + (r+1) + "/" + (i+1));

        // initialize labels or flip ones close to 0.5
        if (firstRun)
          initializeLabels();
        else
          flipLabels();

        // build classifier
        buildClassifier();

        // calc RMS and Accuracy
        calculateRMS();
        calculateAccuracy();
        
        // log internal data
        if (getLog())
          logQuality(r, i);

        // if new model is better than old stored one, then store it
        if (    (m_BestModel == null)
            || (this.compareTo(m_BestModel) > 0) ) {

          if ( (m_BestModel != null) && (getVerbose()) )
            System.out.print("  --> Classifier is better");
          
          m_LastRestart         = r;
          m_LastIteration       = i;
          m_LastRMS             = m_RMS;
          m_LastRMSTrain        = m_RMSTrain;
          m_LastRMSTest         = m_RMSTest;
          m_LastRMSTestOriginal = m_RMSTestOriginal;
          m_LastAccTrain        = m_AccTrain;
          // we don't want to store the previous best model!
          m_BestModel           = null;

          if (    (m_EvaluationType == CollectiveInstances.EVAL_RANDOMWALK_BEST)
               || (m_EvaluationType == CollectiveInstances.EVAL_HILLCLIMBING) )
            m_BestModel = (SimpleCollective) makeCopy(this);
        }
        
        if (getVerbose())
          System.out.println();
        firstRun = false;
      }
    }
  }

  /**
   * generates a filename with a certain suffix for the logQuality method
   * 
   * @param suffix	the suffix
   * @return		the generated filename
   * @see		#logQuality(int,int)
   */
  protected String createFilename(String suffix) {
    String        result;

    result =   this.getClass().getName()
             + "-" + CollectiveHelper.generateMD5(
                        Utils.joinOptions(this.getOptions()))
             + "-" + (m_Trainset == null ? "null" : m_Trainset.relationName().replaceAll("-weka\\.filters\\..*", "")) 
             + "-R" + getNumRestarts()
             + "-I" + getNumIterations()
             + "-E" + m_EvaluationType
             + "-C" + m_ComparisonType
             + suffix;

    return result;
  }

  /**
   * returns the percentage of flipped labels
   * 
   * @return		the percentage
   * @see 		CollectiveInstances#getFlippedLabels()
   */
  protected double getFlippedLabels() {
    return m_CollectiveInstances.getFlippedLabels();
  }

  /**
   * logs some internal quality measures
   * 
   * @param restart	the restart
   * @param iteration	the iteration
   * @throws Exception	if something goes wrong
   */
  protected void logQuality(int restart, int iteration)
    throws Exception {
    
    m_LogEntries.addValue("RMS",             m_RMS);
    m_LogEntries.addValue("RMSTrain",        m_RMSTrain);
    m_LogEntries.addValue("RMSTest",         m_RMSTest);
    m_LogEntries.addValue("RMSTestOriginal", m_RMSTestOriginal);
    m_LogEntries.addValue("AccTrain",        m_AccTrain);
    m_LogEntries.addValue("AccTestOriginal", m_AccTestOriginal);
    m_LogEntries.addValue("FlippedLabels",   getFlippedLabels());
      
    // store it
    if (    (iteration == m_NumIterations - 1) 
         && (!m_LogEntries.getValues("RMS").equals("")) ) {
      m_LogEntries.write();
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
    result += "Last Restart..........: " + (getLastRestart()+1) + "\n";
    result += "Last Iteration........: " + (getLastIteration()+1) + "\n";
    result += "Last RMS..............: " + getLastRMS() + "\n";
    result += "Last RMS (train)......: " + getLastRMSTrain() + "\n";
    result += "Last RMS (test).......: " + getLastRMSTest() + "\n";
    if (getUseInsight())
      result += "Last RMS (orig. test).: " + getLastRMSTestOriginal() + "\n";
    result += "Last Accuracy (train).: " + getLastAccTrain() + "\n";

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
    CollectiveEvaluationUtils.runClassifier(new SimpleCollective(), args);
  }
}
