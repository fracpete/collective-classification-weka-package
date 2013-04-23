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
 * Chopper.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.meta;

import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.collective.CollectiveRandomizableMultipleClassifiersCombiner;
import weka.classifiers.collective.util.CollectiveHelper;
import weka.classifiers.evaluation.CollectiveEvaluationUtils;
import weka.core.AttributeStats;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.Summarizable;
import weka.core.Utils;

/**
 <!-- globalinfo-start -->
 * Collective Classifier that uses one classifier for labeling the test data after training on the train set. The trained classifier determines the distributions for all the instances in the test set and uses the difference between the two confidences (works, therefore, only on two-class-problems) to rank the instances.<br/>
 * The fold with the highest ranking (biggest difference between the two confidences) is then added to the training set - after the labels have been determined.  This new training set is once again input for a classifier (if there are more than two classifiers given, then the classifiers 2 to n are used in turns for training on the newly generated data and classifying the rest of the test data for the next one) which determines once again the distributions for the remaining instances of the test set.
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -F &lt;num&gt;
 *  Number of folds to chop the ranked test instances into.
 *  (default 10)</pre>
 * 
 * <pre> -cut-off &lt;num&gt;
 *  Number of folds after which to stop the adding of folds.
 *  0 means no cut-off at all. (default 0)</pre>
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
 * <pre> -B &lt;classifier specification&gt;
 *  Full class name of classifier to include, followed
 *  by scheme options. May be specified multiple times.
 *  (default: "weka.classifiers.rules.ZeroR")</pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 <!-- options-end -->
 *
 * @author Bernhard Pfahringer (bernhard at cs dot waikato dot ac dot nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $ 
 */
public class Chopper
  extends CollectiveRandomizableMultipleClassifiersCombiner 
  implements Summarizable {
  
  /** for serialization */
  private static final long serialVersionUID = 6356198691065529678L;

  /** the train set to work with */
  protected Instances m_TrainsetNew;
  
  /** the test set to work with */
  protected Instances m_TestsetNew;

  /** the number of folds to chop the ranked test instances into */
  protected int m_Folds;

  /** the number of folds after which to stop the adding of folds */
  protected int m_CutOff;

  /** the current classifier to use for iterating */
  protected int m_CurrentClassifierIndex;

  /** Contains the class distribution of the training set. Used in 
   * <code>distributionForInstance(Instance)</code> 
   * @see #distributionForInstance(Instance) */
  protected double[] m_ClassDistribution;

  /** The number of instances to be added per iteration. */
  protected double m_InstancesPerIteration;

  /**
   * performs initialization of members
   */
  @Override
  protected void initializeMembers() {
    super.initializeMembers();
    
    Classifier[] cls = new Classifier[2];
    cls[0] = new weka.classifiers.trees.J48();
    cls[1] = new weka.classifiers.trees.J48();
    setClassifiers(cls);

    m_TrainsetNew            = null;
    m_TestsetNew             = null;
    m_Folds                  = 10;
    m_CutOff                 = 0;
    m_CurrentClassifierIndex = -1;
    m_ClassDistribution      = null;
    m_InstancesPerIteration  = 0;
  }
  
  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return    
        "Collective Classifier that uses one classifier for labeling the test "
      + "data after training on the train set. The trained classifier "
      + "determines the distributions for all the instances in the test set "
      + "and uses the difference between the two confidences (works, "
      + "therefore, only on two-class-problems) to rank the instances.\n"
      + "The fold with the highest ranking (biggest difference between the "
      + "two confidences) is then added to the training set - after the "
      + "labels have been determined.  This new training set is once again "
      + "input for a classifier (if there are more than two classifiers "
      + "given, then the classifiers 2 to n are used in turns for "
      + "training on the newly generated data and classifying the rest of "
      + "the test data for the next one) which determines once again the "
      + "distributions for the remaining instances of the test set.";
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
        "\tNumber of folds to chop the ranked test instances into.\n"
        + "\t(default 10)",
        "F", 1, "-F <num>"));
    
    result.addElement(new Option(
        "\tNumber of folds after which to stop the adding of folds.\n"
        + "\t0 means no cut-off at all. (default 0)",
        "cut-off", 1, "-cut-off <num>"));
    
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
   * <pre> -F &lt;num&gt;
   *  Number of folds to chop the ranked test instances into.
   *  (default 10)</pre>
   * 
   * <pre> -cut-off &lt;num&gt;
   *  Number of folds after which to stop the adding of folds.
   *  0 means no cut-off at all. (default 0)</pre>
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
   * <pre> -B &lt;classifier specification&gt;
   *  Full class name of classifier to include, followed
   *  by scheme options. May be specified multiple times.
   *  (default: "weka.classifiers.rules.ZeroR")</pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {
    String        	tmpStr;
    Classifier[]	cls;
    
    tmpStr = Utils.getOption('F', options);
    if (tmpStr.length() != 0)
      setFolds(Integer.parseInt(tmpStr));
    else
      setFolds(10);
    
    tmpStr = Utils.getOption("cut-off", options);
    if (tmpStr.length() != 0)
      setCutOff(Integer.parseInt(tmpStr));
    else
      setCutOff(0);
    
    super.setOptions(options);
    
    // fix the classifiers
    if (getClassifiers().length < 2) {
      System.out.println("Less than two classifiers - defaulting to two J48 instances!");
      cls = new Classifier[2];
      cls[0] = new weka.classifiers.trees.J48();
      cls[1] = new weka.classifiers.trees.J48();
      setClassifiers(cls);
    }
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

    result.add("-F");
    result.add("" + getFolds());
    
    result.add("-cut-off");
    result.add("" + getCutOff());
    
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
   * Sets the number of folds to divide the ranked test instances into
   * 
   * @param value	the number of folds
   */
  public void setFolds(int value) {
    if (value > 1)
      m_Folds = value;
    else
      System.out.println("Must have at least 2 folds (provided: " +
        value + ")!");
  }
  
  /**
   * Gets the number of folds the ranked test instances are divided into
   *
   * @return the number of folds
   */
  public int getFolds() {
    return m_Folds;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String foldsTipText() {
    return "The number of folds to divide the ranked test instances into.";
  }
  
  /**
   * Sets the cut-off number for the folds
   * 
   * @param value	the cut-off fold
   */
  public void setCutOff(int value) {
    if (value >= 0)
      m_CutOff = value;
    else
      System.out.println("Cut-Off must be 0 (no cut-off) or greater than 0 "
          + " (provided: " + value + ")!");
  }
  
  /**
   * Gets the fold cut-off
   *
   * @return the cut-off
   */
  public int getCutOff() {
    return m_CutOff;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String cutOffTipText() {
    return "The number of folds after which to stop adding folds to the training set (0 means no cut-off).";
  }
  
  /**
   * resets the classifier
   */
  @Override
  public void reset() {
    super.reset();

    m_CurrentClassifierIndex = -1;
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
    return getCurrentClassifier().distributionForInstance(instance);
  }

  /**
   * returns the next classifier for iterating, in case that there is more
   * than one classifier specified for iterating. Otherwise it always returns
   * the same one.
   * 
   * @return		the next classifier
   * @throws Exception	if something goes wrong
   */
  protected Classifier getNextClassifier() throws Exception {
    m_CurrentClassifierIndex++;
    if (m_CurrentClassifierIndex > getClassifiers().length - 1)
      m_CurrentClassifierIndex = 1;

    return getCurrentClassifier();
  }

  /**
   * returns the current classifier used for iterating.
   * @return      the current classifier used for iterating
   * @see #getNextClassifier()
   */
  protected Classifier getCurrentClassifier() {
    return getClassifiers()[m_CurrentClassifierIndex];
  }
  
  /**
   * performs the actual building of the classifier
   * @throws Exception if building fails
   */
  @Override
  protected void buildClassifier() throws Exception {
    Classifier      c;
    
    c = getNextClassifier();
    if (getDebug())
      System.out.println(
          "buildClassifier: " 
          + m_CurrentClassifierIndex + ". " 
          + c.getClass().getName());
      
    c.buildClassifier(m_TrainsetNew);
  }
  
  /**
   * Method for building this classifier.
   * 
   * @param training	the training instances
   * @param test	the test instances
   * @throws Exception	if seomthing goes wrong, e.g., less than 2 base classifiers
   */
  @Override
  public void buildClassifier(Instances training, Instances test) throws Exception {
    // is there no classifier?
    if (getClassifiers().length < 2)
      throw new Exception(
        "This classifier needs at least 2 base classifier, but instead "
        + getClassifiers().length + " was/were provided!");
   
    super.buildClassifier(training, test);
  }

  /**
   * builds the training set, depending on the current iteration adding the
   * best currentIter/totalFolds * #test-instances to original training set
   * 
   * @param currentIteration	the current iteration
   * @throws Exception		if something goes wrong
   */
  protected void buildTrainSet(int currentIteration) throws Exception {
    int               i;
    double[]          margins;
    double[]          prediction;
    String[]          classLabels;
    int               numberInstances;
    int[]             indices;
    int[]             copiedIndices;
    Instance          inst;
    
    if (currentIteration == 0) {
      m_TrainsetNew = new Instances(m_Trainset);
      m_TestsetNew  = new Instances(m_Testset);
    }
    else {
      // classify all instances and record margins/class labels
      margins     = new double[m_TestsetNew.numInstances()];
      classLabels = new String[m_TestsetNew.numInstances()];
      for (i = 0; i < m_TestsetNew.numInstances(); i++) {
        prediction     = getDistribution(m_TestsetNew.instance(i));
        margins[i]     = StrictMath.abs(prediction[0] - prediction[1]);
        classLabels[i] = m_TestsetNew.classAttribute().value(
                              Utils.maxIndex(prediction));
      }

      // sort margins, determine best instances and add them to the train set
      // (and of course removing them from the test set)
      indices = Utils.stableSort(margins);
      if (currentIteration == getFolds())
        numberInstances = m_TestsetNew.numInstances();
      else
        numberInstances =   
             (int) (m_InstancesPerIteration * currentIteration)
           - (int) (m_InstancesPerIteration * (currentIteration - 1));
      if (getDebug())
        System.out.println("Number of instances to add: " + numberInstances);
      
      // add to train set (and set label)
      copiedIndices = new int[numberInstances];
      for (i = 0; i < numberInstances; i++) {
        copiedIndices[i] = indices[margins.length - (i + 1)];
        inst = (Instance) m_TestsetNew.instance(copiedIndices[i]).copy();
        inst.setClassValue(classLabels[copiedIndices[i]]);
        m_TrainsetNew.add(inst);
      }

      // remove from test set (have to start at the end of sorted array, 
      // otherwise the indices are out of sync!)
      Arrays.sort(copiedIndices);
      for (i = copiedIndices.length - 1; i >= 0; i--)
        m_TestsetNew.delete(copiedIndices[i]);
    }

    // output size of datasets
    if (getDebug())
      System.out.println(  "After adding: Train=" + m_TrainsetNew.numInstances() 
                         + ", Test=" + m_TestsetNew.numInstances() );
  }
  
  /**
   * builds the classifier
   * 
   * @throws Exception	if something goes wrong
   */
  @Override
  protected void build() throws Exception {
    AttributeStats        stats;
    int                   i;
    
    // determine class distribution
    m_ClassDistribution = new double[2];
    stats = m_Trainset.attributeStats(m_Trainset.classIndex());
    for (i = 0; i < 2; i++)
      m_ClassDistribution[i] = stats.nominalCounts[i] / stats.totalCount;

    // the number of instances added to the training set in each iteration
    m_InstancesPerIteration =   (double) m_Testset.numInstances() 
                              / getFolds();
    if (getDebug())
      System.out.println("InstancesPerIteration: " + m_InstancesPerIteration);

    // build classifier
    m_Random = new Random(getSeed());
    for (i = 0; i <= getFolds(); i++) {
      if (getVerbose() || getDebug()) {
        if (getCutOff() > 0)
          System.out.println(   "\nFold " + i + "/" + getFolds() 
                              + " (CutOff at " + getCutOff() + ")");
        else
          System.out.println("\nFold " + i + "/" + getFolds());
      }
      buildTrainSet(i);
      buildClassifier();
      
      // cutoff of folds reached?
      if ( (i > 0) && (i == getCutOff()) )
        break;
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
    int           i;
    int           folds;
    int           index;
    
    if (getClassifiers().length >= 2) {
      result =    "Classifier (init).....: " 
                + getClassifiers()[0].getClass().getName() + "\n";
      for (i = 1; i < getClassifiers().length; i++)
        result +=   "Classifier (train)....: " 
                  + getClassifiers()[i].getClass().getName() + "\n";
      // determine classifier that is used for predictions
      if (getCutOff() > 0)
        folds = getCutOff();
      else
        folds = getFolds();
      index = folds % (getClassifiers().length - 1);
      if (index == 0)
        index = getClassifiers().length - 1;
      result +=   "used for predictions..: " 
                + getClassifiers()[index].getClass().getName() 
                + " (train classifier " + index + ")";
    }
    else {
      result = "";
    }
    
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
    StringBuffer    text;
    int             i;
    
    text = new StringBuffer();
    text.append(super.toStringModel());
    text.append("\n");
    text.append("1. Classifier for initializing the Test-Labels\n");
    text.append("----------------------------------------------\n");
    text.append(getClassifiers()[0].toString());
    text.append("\n");
    for (i = 1; i < getClassifiers().length; i++) {
      text.append("" + (i+1) + ". Classifier for Training\n");
      text.append("--------------------------\n");
      text.append(getClassifiers()[i].toString());
    }
    
    return text.toString();
  }

  /**
   * Returns a string that summarizes the object. We abuse this method to
   * store the MD5 of the options.
   *
   * @return the object summarized as a string
   */
  public String toSummaryString() {
    return CollectiveHelper.generateMD5(Utils.joinOptions(getOptions()));
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
    CollectiveEvaluationUtils.runClassifier(new Chopper(), args);
  }
}
