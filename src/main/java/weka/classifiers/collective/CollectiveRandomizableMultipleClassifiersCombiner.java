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
 * CollectiveRandomizableClassifier.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective;

import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.RandomizableMultipleClassifiersCombiner;
import weka.classifiers.collective.util.Clock;
import weka.classifiers.collective.util.Splitter;
import weka.core.AdditionalMeasureProducer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;

/**
 * Abstract classifier for a randomizable collective classifer combining
 * multiple classifiers.
 * 
 * @author  Fracpete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $ 
 */
public abstract class CollectiveRandomizableMultipleClassifiersCombiner
  extends RandomizableMultipleClassifiersCombiner
  implements CollectiveClassifier, AdditionalMeasureProducer {
  
  /** whether the classifier was already built
   * (<code>buildClassifier(Instances)</code> only stores the training set.
   * Actual training is done in
   * <code>distributionForInstance(Instances)</code>)
   */
  protected boolean m_ClassifierBuilt;
  
  /** whether to output some information during improving the classifier */
  protected boolean m_Verbose;

  /** The test instances */
  protected Instances m_Testset;

  /** The test instances (with the original labels) 
   * @see #m_UseInsight */
  protected Instances m_TestsetOriginal;
  
  /** The training instances */
  protected Instances m_Trainset;
  
  /** Random number generator */
  protected Random m_Random;
  
  /** The number of folds to split the training set into train and test set.
   *  E.g. 5 folds result in 20% train and 80% test set. */
  protected int m_SplitFolds;
  
  /** Whether to invert the folds, i.e., instead of taking the first fold as
   *  training set it is taken as test set and the rest as training. */
  protected boolean m_InvertSplitFolds;

  /** Stores the original labels of the dataset. Used for outputting some more 
   * statistics about the learning process. */
  protected boolean m_UseInsight;

  /** For additional measures */
  protected Vector m_AdditionalMeasures;

  /** for clocking the time */
  protected Clock m_Clock;

  /**
   * initializes the classifier
   */
  public CollectiveRandomizableMultipleClassifiersCombiner() {
    super();
    initializeMembers();
    reset();
  }

  /**
   * performs initialization of members
   */
  protected void initializeMembers() {
    m_ClassifierBuilt    = false;
    m_Verbose            = false;
    m_Testset            = null;
    m_TestsetOriginal    = null;
    m_Trainset           = null;
    m_Random             = null;
    m_SplitFolds         = 0;
    m_InvertSplitFolds   = false;
    m_UseInsight         = false;
    m_AdditionalMeasures = new Vector();
    m_Clock              = new Clock(false);
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
        "\tThe number of folds for splitting the training set into\n"
        + "\ttrain and test set. The first fold is always the training\n"
        + "\tset. With '-V' you can invert this, i.e., instead of 20/80\n"
        + "\tfor 5 folds you'll get 80/20.\n"
        + "\t(default 0 - no splitting, test = train)",
        "folds", 1, "-folds <folds>"));
    
    result.addElement(new Option(
        "\tInverts the fold selection, i.e., instead of using the first\n"
        + "\tfold for the training set it is used for test set and the\n"
        + "\tremaining folds for training.",
        "V", 0, "-V"));
    
    result.addElement(new Option(
        "\tWhether to print some more information during building the\n"
        + "\tclassifier.\n"
        + "\t(default is off)",
        "verbose", 0, "-verbose"));
    
    result.addElement(new Option(
        "\tWhether to print some more information during building the\n"
        + "\tclassifier.\n"
        + "\t(default is off)",
        "verbose", 0, "-verbose"));
    
    Enumeration en = super.listOptions();
    while (en.hasMoreElements())
      result.addElement(en.nextElement());

    return result.elements();
  }
  
  /**
   * Parses a given list of options. Valid options are:<p/>
   *
   * -folds folds <br/>
   * the number of folds for splitting the training set into train and test
   * set.  the first fold is always the training set. With '-V' you can invert
   * this, i.e., instead of 20/80 for 5 folds you'll get 80/20. (default 5) <p/>
   *
   * -V <br/>
   * inverts the fold selection, i.e., instead of using the first fold for the
   * training set it is used for test set and the remaining folds for training.
   * <p/>
   *
   * -verbose <br/> whether to output some more information during improving the
   *  classifier. <p/>
   *
   * -insight <br/> 
   *  whether to use the labels of the original test set to output more
   *  statistics. <p/>
   *
   * Options after -- are passed to the designated classifier.<p/>
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {
    String        tmpStr;

    tmpStr = Utils.getOption("folds", options);
    if (tmpStr.length() != 0)
      setSplitFolds(Integer.parseInt(tmpStr));
    else
      setSplitFolds(0);
    
    setInvertSplitFolds(Utils.getFlag('V', options));
 
    setVerbose(Utils.getFlag("verbose", options));
    
    setUseInsight(Utils.getFlag("insight", options));
    
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

    result.add("-folds");
    result.add("" + getSplitFolds());
    
    if (getInvertSplitFolds())
      result.add("-V");

    if (getVerbose())
      result.add("-verbose");
    
    if (getUseInsight())
      result.add("-insight");
    
    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);
    
    return (String[]) result.toArray(new String[result.size()]);
  }

  /**
   * Returns an enumeration of the measure names. Additional measures
   * must follow the naming convention of starting with "measure", eg.
   * double measureBlah()
   * @return an enumeration of the measure names
   */
  public Enumeration enumerateMeasures() {
    return m_AdditionalMeasures.elements();
  }

  /**
   * Returns the value of the named measure
   * @param measureName the name of the measure to query for its value
   * @return the value of the named measure
   * @throws IllegalArgumentException if the named measure is not supported
   */
  public double getMeasure(String measureName) {
    throw new IllegalArgumentException(
          measureName 
        + " not supported (" 
        + this.getClass().getName().replaceAll(".*\\.", "")
        + ")");
  }
  
  /**
   * sets the instances used for testing
   * 
   * @param value the instances used for testing
   */
  public void setTestSet(Instances value) {
    m_ClassifierBuilt = false;
    m_Testset         = value;
    if (getUseInsight())
      m_TestsetOriginal = new Instances(value);
    else
      m_TestsetOriginal = null;
  }
  
  /**
   * returns the Test Set
   *
   * @return the Test Set
   */
  public Instances getTestSet() {
    return m_Testset;
  }
  
  /**
   * returns the Training Set 
   *
   * @return the Training Set
   */
  public Instances getTrainingSet() {
    return m_Trainset;
  }
  
  /**
   * Set the verbose state.
   *
   * @param value the verbose state
   */
  public void setVerbose(boolean value) {
    m_Verbose = value;
  }
  
  /**
   * Gets the verbose state
   *
   * @return the verbose state
   */
  public boolean getVerbose() {
    return m_Verbose;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String verboseTipText() {
    return "Whether to ouput some additional information during building.";
  }
  
  /**
   * Set the percentage for splitting the train set into train and test set.
   * Use 0 for no splitting, which results in test = train.
   *
   * @param value the split percentage (1/splitFolds*100)
   */
  public void setSplitFolds(int value) {
    if (value >= 2)
      m_SplitFolds = value;
    else
      m_SplitFolds = 0;
  }
  
  /**
   * Gets the split percentage for splitting train set into train and test set
   *
   * @return the split percentage (1/splitFolds*100)
   */
  public int getSplitFolds() {
    return m_SplitFolds;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String splitFoldsTipText() {
    return "The percentage to use for splitting the train set into train and test set.";
  }
  
  /**
   * Sets whether to use the first fold as training set (= FALSE) or as
   * test set (= TRUE).
   *
   * @param value whether to invert the folding scheme
   */
  public void setInvertSplitFolds(boolean value) {
    m_InvertSplitFolds = value;
  }
  
  /**
   * Gets whether to use the first fold as training set (= FALSE) or as
   * test set (= TRUE)
   *
   * @return whether to invert the folding scheme
   */
  public boolean getInvertSplitFolds() {
    return m_InvertSplitFolds;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String invertSplitFoldsTipText() {
    return "Whether to invert the folding scheme.";
  }

  /**
   * Whether to use the labels of the test set to output more statistics
   * (not used for learning and only for debugging purposes)
   * 
   * @param value	if true, more statistics are output
   */
  public void setUseInsight(boolean value) {
    m_UseInsight = value;
  }

  /**
   * Returns whether we use the labels of the test set to output some more
   * statistics.
   * 
   * @return		true if more statistics are output
   */
  public boolean getUseInsight() {
    return m_UseInsight;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String useInsightTipText() {
    return "Whether to use the original labels of the test set to generate more statistics (not used for learning!).";
  }

  /**
   * returns the specification of the given object (class, options if an 
   * instance of OptionHandler)
   *
   * @param o		the object to get the specs as string
   * @return		the specification string
   */
  protected String getSpecification(Object o) {
    String      result;

    result = o.getClass().getName();
    if (o instanceof OptionHandler)
      result += " " + Utils.joinOptions(((OptionHandler) o).getOptions());

    return result.trim();
  }

  /**
   * starts the clock
   */
  protected void clock() {
    m_Clock.start();
  }

  /**
   * stops the clock and prints the given msg, followed by the time, if in
   * debug mode
   * 
   * @param msg		the message to print
   */
  protected void clock(String msg) {
    m_Clock.stop();

    if (getVerbose())
      System.out.println(msg + ": " + m_Clock);
  }
  
  /**
   * splits the train set into train and test set if no test set was provided,
   * according to the set SplitFolds.
   *
   * @see #getSplitFolds()
   * @see #getInvertSplitFolds()
   * @throws Exception if anything goes wrong with the Filter
   */
  protected void splitTrainSet() throws Exception {
    Splitter        splitter;

    splitter = new Splitter(m_Trainset);
    splitter.setSplitFolds(getSplitFolds());
    splitter.setInvertSplitFolds(getInvertSplitFolds());
    splitter.setVerbose(getVerbose());
    
    m_Trainset = splitter.getTrainset();
    m_Testset  = splitter.getTestset();
  }
  
  /**
   * checks whether the classifier was build and if not performs the build (but 
   * only if the testset is not <code>null</code>).
   * 
   * @see 		#m_ClassifierBuilt 
   * @return		true if the classifier was built
   */
  protected boolean checkBuiltStatus() {
    boolean result = m_ClassifierBuilt;
    
    if ( (!result) && (m_Testset != null) ) {
      try {
        buildClassifier(m_Trainset, m_Testset);
        result = m_ClassifierBuilt;
      }
      catch (Exception e) {
        throw new IllegalStateException(e);
      }
    }
    
    return result;
  }
  
  /**
   * Predicts the class memberships for a given instance. If
   * an instance is unclassified, the returned array elements
   * must be all zero. If the class is numeric, the array
   * must consist of only one element, which contains the
   * predicted value. Note that a classifier MUST implement
   * either this or classifyInstance().<br/>
   * Note: if a derived class should override this method, make
   * sure it calls <code>checkBuiltStatus()</code>.
   *
   * @param instance      the instance to be classified
   * @return              an array containing the estimated membership 
   *                      probabilities of the test instance in each class 
   *                      or the numeric prediction
   * @throws Exception if distribution could not be 
   *                      computed successfully
   * @see #checkBuiltStatus()
   */
  @Override
  public double[] distributionForInstance(Instance instance) throws Exception {
    // no testset if after buildClassifier(Instances) call 
    // this method is -> split training set
    // e.g., in the Explorer ("Classify")
    if (m_Testset == null) {
      splitTrainSet();
      generateSets();
    }
   
    checkBuiltStatus();
    
    return getDistribution(instance); 
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
  protected abstract double[] getDistribution(Instance instance) 
    throws Exception;
  
  /**
   * resets the classifier
   */
  public void reset() {
    if (getDebug() || getVerbose())
      System.out.println("Reseting the classifier...");
    m_ClassifierBuilt = false;
  }

  /**
   * Checks the data, whether it can be used. If not Exceptions are thrown
   * @throws Exception if the data doesn't fit in any way
   */
  protected void checkData() throws Exception {
    if (m_Testset == null)
      throw new Exception("No Test instances provided!");
    
    if (!m_Trainset.equalHeaders(m_Testset))
      throw new Exception("Training and Test Set not compatible!");
  }
  
  /**
   * the standard collective classifier accepts only nominal, binary classes
   * otherwise an exception is thrown
   * @throws Exception if the data doesn't have a nominal, binary class
   */
  protected void checkRestrictions() throws Exception {
    // can classifier handle the data?
    getCapabilities().testWithFail(m_Trainset);

    // remove instances with missing class
    m_Trainset = new Instances(m_Trainset);
    m_Trainset.deleteWithMissingClass();
    if (m_Testset != null)
      m_Testset = new Instances(m_Testset);
  }
  
  /**
   * builds the necessary datasets from the given Instances
   * 
   * @throws Exception if anything goes wrong
   */
  protected void generateSets() throws Exception {
    Instances         instances;
    Instance          instance;
    
    instances = new Instances(m_Testset, 0);

    for (int i = 0; i < m_Testset.numInstances(); i++) {
      instance = (Instance) m_Testset.instance(i).copy();
      instance.setClassMissing();
      instances.add(instance);
    }

    m_Testset = instances;
  }
  
  /**
   * performs the actual building of the classifier
   * 
   * @throws Exception if building fails
   */
  protected abstract void buildClassifier() throws Exception;

  /**
   * here initialization and building, possible iterations will happen
   * 
   * @throws Exception	if something goes wrong
   */
  protected abstract void build() throws Exception;
  
  /**
   * Method for building this classifier. Since the collective classifiers
   * also need the test set, we only store here the training set.  
   * 
   * @param training        the training set to use
   * @throws Exception      derived classes may throw Exceptions
   */
  public void buildClassifier(Instances training) throws Exception {
    m_ClassifierBuilt = false;
    m_Trainset        = training;

    // set class index?
    if (m_Trainset.classIndex() == -1)
      m_Trainset.setClassIndex(m_Trainset.numAttributes() - 1);

    // necessary for JUnit tests
    checkRestrictions();
  }
  
  /**
   * Method for building this classifier.
   * 
   * @param training	the training instances
   * @param test	the test instances
   * @throws Exception	if something goes wrong
   */
  public void buildClassifier(Instances training, Instances test) throws Exception {
    m_ClassifierBuilt = true;
    m_Random          = new Random(m_Seed);
    m_Trainset        = training;
    m_Testset         = test;

    // set class index?
    if ( (m_Trainset.classIndex() == -1) || (m_Testset.classIndex() == -1) ) {
      m_Trainset.setClassIndex(m_Trainset.numAttributes() - 1);
      m_Testset.setClassIndex(m_Trainset.numAttributes() - 1);
    }

    // are datasets correct?
    checkData();

    // any other data restrictions not met?
    checkRestrictions();
    
    // generate sets
    generateSets();
    
    // performs the restarts/iterations
    build();
    
    m_Random = null;
  }
  
  /**
   * returns information about the classifier(s)
   * 
   * @return		information about the classifier
   */
  protected String toStringClassifier() {
    String        result;
    int           i;

    result = "";
    for (i = 0; i < getClassifiers().length; i++)
      result +=   Integer.toString(i+1) + ". Classifier..........: " 
                + getSpecification(getClassifiers()[i]) + "\n";

    return result;
  }
  
  /**
   * returns some information about the parameters
   * 
   * @return		information about the parameters
   */
  protected String toStringParameters() {
    return "";
  }
  
  /**
   * returns the best model as string representation. derived classes have to 
   * add additional information here, like printing the classifier etc.
   * 
   * @return		the string representation of the best model
   */
  protected String toStringModel() {
    return "Best model printed below...\n";
  }
  
  /**
   * Returns description of the classifier.<br/>
   * Note: if a derived class overrides this method, make sure it calls
   * <code>checkBuiltStatus()</code> to assure a model has been built.
   *
   * @return description of the classifier as a string
   * @see #checkBuiltStatus()
   */
  @Override
  public String toString() {
    StringBuffer    text;
    String          classifier;
    
    text       = new StringBuffer();
    classifier = getClass().getName().replaceAll(".*\\.", "");
    
    text.append(classifier + "\n" + classifier.replaceAll(".", "-"));
    text.append("\n\n");
    text.append(toStringClassifier());
    text.append(toStringParameters());
    text.append("\n");
    if (!checkBuiltStatus())
      text.append("No Test set provided so far, hence no model built yet! See below for model...");
    else
      text.append(toStringModel());
    
    return text.toString();
  }
}
