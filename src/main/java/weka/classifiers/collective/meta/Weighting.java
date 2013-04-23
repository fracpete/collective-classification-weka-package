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
 * Weighting.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.meta;

import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.collective.CollectiveRandomizableMultipleClassifiersCombiner;
import weka.classifiers.collective.util.CollectiveHelper;
import weka.classifiers.evaluation.CollectiveEvaluationUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.Summarizable;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

/**
 <!-- globalinfo-start -->
 * Collective Classifier that uses one classifier for labeling the test data after training on the train set. The trained classifier determines the class labels for all the instances in the test dataset.  This is again input for another classifier (if there are more than two classifiers defined, then the classifiers 2 to n are used in turns).  In the initializing step, all instances from the test set have a weight of 0. In each following step, they get a weight of current_step / number_of_steps. This implies that all provided classifiers need to be able to handle weighted instances.<br/>
 * It also possible to stop this process with the cut-off parameter, which defines the step after which to stop.<br/>
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -steps &lt;num&gt;
 *  Number of steps for the weighting process.
 *  (default 10)</pre>
 * 
 * <pre> -cut-off &lt;num&gt;
 *  Number of steps after which to stop the learning process.
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
public class Weighting
  extends CollectiveRandomizableMultipleClassifiersCombiner 
  implements Summarizable {
  
  /** for serialization */
  private static final long serialVersionUID = -651457447696365135L;

  /** the train set to work with */
  protected Instances m_TrainsetNew;

  /** the number of steps to use for weighting */
  protected int m_Steps;

  /** the number of steps after which to stop the learning */
  protected int m_CutOff;

  /** the current classifier to use for iterating */
  protected int m_CurrentClassifierIndex;

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
    m_Steps                  = 10;
    m_CutOff                 = 0;
    m_CurrentClassifierIndex = -1;
  }
  
  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return    
        "Collective Classifier that uses one classifier for labeling the "
      + "test data after training on the train set. The trained "
      + "classifier determines the class labels for all the instances in "
      + "the test dataset.  This is again input for another classifier "
      + "(if there are more than two classifiers defined, then the "
      + "classifiers 2 to n are used in turns).  In the initializing "
      + "step, all instances from the test set have a weight of 0. In "
      + "each following step, they get a weight of current_step / "
      + "number_of_steps. This implies that all provided classifiers need "
      + "to be able to handle weighted instances.\n"
      + "It also possible to stop this process with the cut-off parameter, "
      + "which defines the step after which to stop.\n";
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
        "\tNumber of steps for the weighting process.\n"
        + "\t(default 10)",
        "steps", 1, "-steps <num>"));
    
    result.addElement(new Option(
        "\tNumber of steps after which to stop the learning process.\n"
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
   * <pre> -steps &lt;num&gt;
   *  Number of steps for the weighting process.
   *  (default 10)</pre>
   * 
   * <pre> -cut-off &lt;num&gt;
   *  Number of steps after which to stop the learning process.
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
    String		tmpStr;
    Classifier[]	cls;
    
    tmpStr = Utils.getOption("steps", options);
    if (tmpStr.length() != 0)
      setSteps(Integer.parseInt(tmpStr));
    else
      setSteps(10);
    
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

    result.add("-steps");
    result.add("" + getSteps());
    
    result.add("-cut-off");
    result.add("" + getCutOff());
    
    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);
    
    return (String[]) result.toArray(new String[result.size()]);
  }
  
  /**
   * Sets the number of folds to divide the ranked test instances into
   * 
   * @param value	the number folds
   */
  public void setSteps(int value) {
    if (value > 1)
      m_Steps = value;
    else
      System.out.println("Must have at least 1 step (provided: " +
        value + ")!");
  }
  
  /**
   * Gets the number of folds the ranked test instances are divided into
   *
   * @return the number of folds
   */
  public int getSteps() {
    return m_Steps;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String stepsTipText() {
    return "The number of steps to use for the learning process.";
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
    return "The number of steps after which to stop the learning process (0 means no cut-off).";
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
   * the standard collective classifier accepts only nominal, binary classes
   * otherwise an exception is thrown. Additionally, all classifiers must be
   * able to handle weighted instances.
   * @throws Exception if the data doesn't have a nominal, binary class
   */
  @Override
  protected void checkRestrictions() throws Exception {
    int         i;
    String      nonWeighted;
    
    super.checkRestrictions();

    // do all implement the WeightedInstancesHandler?
    nonWeighted = "";
    for (i = 0; i < getClassifiers().length; i++) {
      if (!(getClassifiers()[i] instanceof WeightedInstancesHandler)) {
        if (nonWeighted.length() > 0)
          nonWeighted += ", ";
        nonWeighted += getClassifiers()[i].getClass().getName();
      }
    }
    if (nonWeighted.length() > 0)
      throw new Exception(
          "The following classifier(s) cannot handle weighted instances:\n" 
          + nonWeighted);
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
   * @throws Exception	if something goes wrong
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
   * builds the training set, depending on the current iteration setting the
   * weight to (1.0 / m_Steps * currentIteration)
   * 
   * @param currentIteration	the current iteration
   * @throws Exception		if something goes wrong
   */
  protected void buildTrainSet(int currentIteration) throws Exception {
    int           i;
    double        weight;
    int           index;
    double        clsValue;

    weight        = 1.0 / getSteps() * currentIteration;
    if (getDebug())
      System.out.println("current weight for test instances: " + weight);

    m_TrainsetNew = new Instances(m_Trainset);
    for (i = 0; i < m_Testset.numInstances(); i++) {
      m_TrainsetNew.add(m_Testset.instance(i));
      index = m_TrainsetNew.numInstances() - 1;
      m_TrainsetNew.instance(index).setWeight(weight);
      // set label with previously built classifier
      if (currentIteration > 0) {
        clsValue = getCurrentClassifier().classifyInstance(
                      m_TrainsetNew.instance(index));
        m_TrainsetNew.instance(index).setClassValue(clsValue);
      }
    }
  }
  
  /**
   * builds the classifier
   * 
   * @throws Exception 	if something goes wrong
   */
  @Override
  protected void build() throws Exception {
    int       i;
    
    if (getDebug())
      System.out.println("Weight increment per Step: "+(1.0/m_Steps));

    // build classifier
    m_Random = new Random(getSeed());
    for (i = 0; i <= getSteps(); i++) {
      if (getVerbose() || getDebug()) {
        if (getCutOff() > 0)
          System.out.println(   "\nStep " + i + "/" + getSteps() 
                              + " (CutOff at " + getCutOff() + ")");
        else
          System.out.println("\nStep " + i + "/" + getSteps());
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
    int           steps;
    int           index;
    
    if (getClassifiers().length >= 1) {
      result =    "Classifier (init).....: " 
                + getClassifiers()[0].getClass().getName() + "\n";
      for (i = 1; i < getClassifiers().length; i++)
        result +=   "Classifier (train)....: " 
                  + getClassifiers()[i].getClass().getName() + "\n";
      // determine classifier that is used for predictions
      if (getCutOff() > 0)
        steps = getCutOff();
      else
        steps = getSteps();
      index = steps % (getClassifiers().length - 1);
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
   * @return		the string representation of the model
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
    CollectiveEvaluationUtils.runClassifier(new Weighting(), args);
  }
}
