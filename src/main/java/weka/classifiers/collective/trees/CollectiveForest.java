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
 * CollectiveForest.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.trees;

import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.collective.CollectiveRandomizableClassifier;
import weka.classifiers.collective.util.RankedList;
import weka.classifiers.evaluation.CollectiveEvaluationUtils;
import weka.classifiers.trees.RandomTree;
import weka.core.AdditionalMeasureProducer;
import weka.core.AttributeStats;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
/**
 <!-- globalinfo-start -->
 * This collective Classifier uses RandomTrees to build predictions on the test set. It divides the test set into folds and successively adds the test instances with the best predictions to the training set.<br/>
 * The first iteration trains solely on the training set and determines the distributions for all the instances in the test set. From these predictions the best are chosen (this number is the same as the number of instances in a fold).<br/>
 * From then on, the classifier is trained with the training file from the previous run plus the determined best instances during the previous iteration.<br/>
 * If the folds number is 0 then one is basically using a RandomForest.<br/>
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- options-start -->
 * Valid options are: <p/>
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
 * <pre> -insight
 *  Whether to use the labels of the original test set for more
 *  statistics (not used for learning!).
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
 * <pre> -I &lt;number of trees&gt;
 *  Number of trees to build.</pre>
 * 
 * <pre> -K &lt;number of attributes&gt;
 *  Number of attributes to randomly investigate
 *  (&lt;1 = int(log(#attributes)+1)).
 *  (default is 0)</pre>
 * 
 * <pre> -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf.
 *  (default is 1)</pre>
 * 
 * <pre> -F &lt;num&gt;
 *  Number of fold for incremental building.
 *  (default is 2)</pre>
 * 
 * <pre> -cut-off &lt;num&gt;
 *  The number of folds after which the learning process
 *  is stopped - 0 means all folds are executed.
 *  (default is 0)</pre>
 * 
 * <pre> -B
 *  Whether to use bagging instead.</pre>
 * 
 * <pre> -bag-size
 *  The size of the bag, where 0 stands for the 
 *  complete training set, less than zero for a 
 *  percentage of instances (from &lt;0 to -1) and 
 *  greater than zero an absolute number of instances.
 *  (default is 0)</pre>
 * 
 <!-- options-end -->
 *
 * @author Bernhard Pfahringer (bernhard at cs dot waikato dot ac dot nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $
 * @see RandomTree
 */
public class CollectiveForest
  extends CollectiveRandomizableClassifier {

  /** for serialization */
  private static final long serialVersionUID = -7060453502813693278L;

  /** Number of trees in forest. */
  protected int m_NumTrees;

  /** Number of features to consider in random feature selection.
      If less than 1 will use int(logM+1) ) */
  protected int m_NumFeatures;

  /** The actual number of features used in the random feature selection. */
  protected int m_KValue;
    
  /** Minimum number of instances for leaf. */
  protected double m_MinNum;

  /** The number of folds used in incremental building, if set to 0 it behaves
   *  like a RandomForest.
   *  @see weka.classifiers.trees.RandomForest */
  protected int m_Folds;

  /** Whether to use bagging in the process of building. */
  protected boolean m_UseBagging;

  /** Represents the bag size. Less than zero means percentage of training set,
   * greater than zero means absolut number of instances. */
  protected double m_BagSize;

  /** The number of folds after which the learning process is stopped. */
  protected int m_CutOff;

  /** Contains the class distribution of the training set. Used in 
   * <code>distributionForInstance(Instance)</code> 
   * @see #distributionForInstance(Instance) */
  protected double[] m_ClassDistribution;

  /** The training set, enriched by additional labeled data from the test set */
  protected Instances m_TrainsetNew;

  /** The test set, getting smaller with each fold. The training set is "fed"
   * with instances from this set. */
  protected Instances m_TestsetNew;

  /** The number of instances to be added per iteration. */
  protected double m_InstancesPerIteration;

  /** The list of sorted Test instances with their associated distributions. */
  protected RankedList m_List;

  /** The out of bag error that has been calculated */
  protected double m_OutOfBagError;  

  /** The number of pure training nodes that were prevented */
  protected double m_PureTrainNodes;

  /** The number of pure test nodes that were prevented */
  protected double m_PureTestNodes;

  /**
   * performs initialization of members
   */
  @Override
  protected void initializeMembers() {
    super.initializeMembers();

    m_NumTrees              = 100;
    m_NumFeatures           = 0;
    m_KValue                = 0;
    m_MinNum                = 1.0;
    m_Folds                 = 2;
    m_UseBagging            = false;
    m_BagSize               = 0;
    m_CutOff                = 0;
    m_ClassDistribution     = null;
    m_TrainsetNew           = null;
    m_TestsetNew            = null;
    m_InstancesPerIteration = 0;
    m_List                  = null;
    m_OutOfBagError         = 0.0;
    m_PureTrainNodes        = 0;
    m_PureTestNodes         = 0;
    
    m_AdditionalMeasures.add("measureOutOfBagError");
    m_AdditionalMeasures.add("measureKValue");
    m_AdditionalMeasures.add("measurePureNodes");
    m_AdditionalMeasures.add("measurePureTrainNodes");
    m_AdditionalMeasures.add("measurePureTestNodes");
  }

  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return   "This collective Classifier uses RandomTrees to build "
           + "predictions on the test set. It divides the test set "
           + "into folds and successively adds the test instances "
           + "with the best predictions to the training set.\n"
           + "The first iteration trains solely on the training set "
           + "and determines the distributions for all the instances "
           + "in the test set. From these predictions the best are "
           + "chosen (this number is the same as the number of "
           + "instances in a fold).\n"
           + "From then on, the classifier is trained with the "
           + "training file from the previous run plus the determined "
           + "best instances during the previous iteration.\n"
           + "If the folds number is 0 then one is basically using a "
           + "RandomForest.\n";
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String numTreesTipText() {
    return "The number of trees to be generated.";
  }

  /**
   * Get the value of numTrees.
   *
   * @return Value of numTrees.
   */
  public int getNumTrees() {
    return m_NumTrees;
  }
  
  /**
   * Set the value of numTrees.
   *
   * @param newNumTrees Value to assign to numTrees.
   */
  public void setNumTrees(int newNumTrees) {
    m_NumTrees = newNumTrees;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String numFeaturesTipText() {
    return "The number of attributes to be used in random selection (see RandomTree).";
  }

  /**
   * Get the number of features used in random selection.
   *
   * @return Value of numFeatures.
   */
  public int getNumFeatures() {
    return m_NumFeatures;
  }
  
  /**
   * Set the number of features to use in random selection.
   *
   * @param newNumFeatures Value to assign to numFeatures.
   */
  public void setNumFeatures(int newNumFeatures) {
    m_NumFeatures = newNumFeatures;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String minNumTipText() {
    return "The minimum total weight of the instances in a leaf.";
  }

  /**
   * Get the value of MinNum.
   *
   * @return Value of MinNum.
   */
  public double getMinNum() {
    return m_MinNum;
  }
  
  /**
   * Set the value of MinNum.
   *
   * @param newMinNum Value to assign to MinNum.
   */
  public void setMinNum(double newMinNum) {
    if (newMinNum >= 1.0)
      m_MinNum = newMinNum;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String foldsTipText() {
    return "The number of folds used for incremental building.";
  }

  /**
   * Get the value of folds.
   *
   * @return Value of folds.
   */
  public int getFolds() {
    return m_Folds;
  }
  
  /**
   * Set the value of folds.
   *
   * @param newFolds Value to assign to folds.
   */
  public void setFolds(int newFolds) {
    if (newFolds >= 0)
      m_Folds = newFolds;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String cutOffTipText() {
    return "The number of folds after which the learning process is stopped.";
  }

  /**
   * Get the fold cutoff value.
   *
   * @return Value of fold cutoff.
   */
  public int getCutOff() {
    return m_CutOff;
  }
  
  /**
   * Set the value for fold cutoff.
   *
   * @param newCutOff Value to assign to fold cutoff.
   */
  public void setCutOff(int newCutOff) {
    if (newCutOff >= 0)
      m_CutOff = newCutOff;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String useBaggingTipText() {
    return "Whether to use bagging in the building process.";
  }

  /**
   * Gets whether bagging is used.
   *
   * @return Whether bagging is used.
   */
  public boolean getUseBagging() {
    return m_UseBagging;
  }
  
  /**
   * Sets whether bagging is used.
   *
   * @param newUseBagging Whether bagging is used.
   */
  public void setUseBagging(boolean newUseBagging) {
    m_UseBagging = newUseBagging;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String bagSizeTipText() {
    return   "The size of the bag, where 0 stands for the complete training "
           + "set, less than zero for a percentage of instances (from <0 to -1) "
           + "and greater than zero an absolute number of instances.";
  }

  /**
   * Get the size of the bag.
   *
   * @return Size of the bag.
   */
  public double getBagSize() {
    return m_BagSize;
  }
  
  /**
   * Set the size of the bag.
   *
   * @param newBagSize The new size of the bag.
   */
  public void setBagSize(double newBagSize) {
    m_BagSize = newBagSize;
  }
  
  /**
   * Lists the command-line options for this classifier.
   * 
   * @return	all the available options
   */
  @Override
  public Enumeration listOptions() {
    Vector        result;
    Enumeration   en;

    result = new Vector();
    en     = super.listOptions();
    while (en.hasMoreElements())
      result.addElement(en.nextElement());
    
    result.addElement(
      new Option("\tNumber of trees to build.",
                 "I", 1, "-I <number of trees>"));
     
    result.addElement(
      new Option(  "\tNumber of attributes to randomly investigate\n"
                 + "\t(<1 = int(log(#attributes)+1)).\n"
                 + "\t(default is 0)",
                 "K", 1, "-K <number of attributes>"));

    result.addElement(
      new Option(  "\tSet minimum number of instances per leaf.\n"
                 + "\t(default is 1)",
                 "M", 1, "-M <minimum number of instances>"));

    result.addElement(
      new Option(  "\tNumber of fold for incremental building.\n"
                 + "\t(default is 2)",
                 "F", 1, "-F <num>"));
     
    result.addElement(
      new Option(  "\tThe number of folds after which the learning process\n"
                 + "\tis stopped - 0 means all folds are executed.\n"
                 + "\t(default is 0)",
                 "cut-off", 1, "-cut-off <num>"));
     
    result.addElement(
      new Option("\tWhether to use bagging instead.",
                 "B", 0, "-B"));
     
    result.addElement(
      new Option(  "\tThe size of the bag, where 0 stands for the \n"
                 + "\tcomplete training set, less than zero for a \n"
                 + "\tpercentage of instances (from <0 to -1) and \n"
                 + "\tgreater than zero an absolute number of instances.\n"
                 + "\t(default is 0)",
                 "bag-size", 0, "-bag-size"));
     
    return result.elements();
  } 

  /**
   * Gets options from this classifier.
   * 
   * @return		the options for the current setup
   */
  @Override
  public String[] getOptions() {
    Vector        result;
    String[]      options;
    int           i;
    
    result  = new Vector();
    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);
    
    result.add("-I");
    result.add("" + getNumTrees());
    
    result.add("-K");
    result.add("" + getNumFeatures());
        
    result.add("-M");
    result.add("" + getMinNum());
      
    result.add("-F");
    result.add("" + getFolds());
      
    result.add("-cut-off");
    result.add("" + getCutOff());
      
    if (getUseBagging()) {
      result.add("-B");
      result.add("-bag-size");
      result.add("" + getBagSize());
    }
    
    return (String[]) result.toArray(new String[result.size()]);
  }

  /**
   * Parses a given list of options. <p/>
   * 
   <!-- options-start -->
   * Valid options are: <p/>
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
   * <pre> -insight
   *  Whether to use the labels of the original test set for more
   *  statistics (not used for learning!).
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
   * <pre> -I &lt;number of trees&gt;
   *  Number of trees to build.</pre>
   * 
   * <pre> -K &lt;number of attributes&gt;
   *  Number of attributes to randomly investigate
   *  (&lt;1 = int(log(#attributes)+1)).
   *  (default is 0)</pre>
   * 
   * <pre> -M &lt;minimum number of instances&gt;
   *  Set minimum number of instances per leaf.
   *  (default is 1)</pre>
   * 
   * <pre> -F &lt;num&gt;
   *  Number of fold for incremental building.
   *  (default is 2)</pre>
   * 
   * <pre> -cut-off &lt;num&gt;
   *  The number of folds after which the learning process
   *  is stopped - 0 means all folds are executed.
   *  (default is 0)</pre>
   * 
   * <pre> -B
   *  Whether to use bagging instead.</pre>
   * 
   * <pre> -bag-size
   *  The size of the bag, where 0 stands for the 
   *  complete training set, less than zero for a 
   *  percentage of instances (from &lt;0 to -1) and 
   *  greater than zero an absolute number of instances.
   *  (default is 0)</pre>
   * 
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {
    String        tmpStr;
    
    super.setOptions(options);
   
    tmpStr = Utils.getOption('I', options);
    if (tmpStr.length() != 0)
      setNumTrees(Integer.parseInt(tmpStr));
    else
      setNumTrees(100);
    
    tmpStr = Utils.getOption('K', options);
    if (tmpStr.length() != 0)
      setNumFeatures(Integer.parseInt(tmpStr));
    else
      setNumFeatures(0);
    
    tmpStr = Utils.getOption('M', options);
    if (tmpStr.length() != 0)
      setMinNum(Double.parseDouble(tmpStr));
    else
      setMinNum(1);
    
    tmpStr = Utils.getOption('F', options);
    if (tmpStr.length() != 0)
      setFolds(Integer.parseInt(tmpStr));
    else
      setFolds(2);
    
    tmpStr = Utils.getOption("cut-off", options);
    if (tmpStr.length() != 0)
      setCutOff(Integer.parseInt(tmpStr));
    else
      setCutOff(0);
    
    setUseBagging(Utils.getFlag('B', options));
    
    tmpStr = Utils.getOption("bag-size", options);
    if (tmpStr.length() != 0)
      setBagSize(Double.parseDouble(tmpStr));
    else
      setBagSize(0);
  }

  /**
   * Returns the value of the named measure
   * @param measureName the name of the measure to query for its value
   * @return the value of the named measure
   * @throws IllegalArgumentException if the named measure is not supported
   */
  @Override
  public double getMeasure(String measureName) {
    if (measureName.equalsIgnoreCase("measureOutOfBagError")) 
      return m_OutOfBagError;
    else if (measureName.equalsIgnoreCase("measureKValue"))
      return m_KValue;
    else if (measureName.equalsIgnoreCase("measurePureNodes"))
      return m_PureTrainNodes + m_PureTestNodes;
    else if (measureName.equalsIgnoreCase("measurePureTrainNodes"))
      return m_PureTrainNodes;
    else if (measureName.equalsIgnoreCase("measurePureTestNodes"))
      return m_PureTestNodes;
    else
      return super.getMeasure(measureName);
  }
  
  /**
   * internal function for determining the class distribution for an instance, 
   * will be overridden by derived classes. For more details about the returned 
   * array, see <code>Classifier.distributionForInstance(Instance)</code>.
   * 
   * @param instance	the instance to get the distribution for
   * @return		the distribution for the given instance
   * @see		Classifier#distributionForInstance(Instance)
   * @throws Exception 	if something goes wrong
   */
  @Override
  protected double[] getDistribution(Instance instance) 
    throws Exception {
    
    return m_List.getDistribution(instance);
  }

  /**
   * calculates the bag size depending on the dataset
   * 
   * @param data	the data to base the calculation on
   * @return		the calculated bag size
   */
  protected int calcBagSize(Instances data) {
    int       result;
    
    if (getBagSize() == 0.0)
      result = data.numInstances();
    else if (getBagSize() < 0)
      result = (int) StrictMath.ceil(-getBagSize() * data.numInstances());
    else
      result = (int) StrictMath.round(getBagSize());

    return result;
  }

  /**
   * generates a new dataset from the 
   * 
   * @param data	the data to work with
   * @param seed	the seed value for resampling
   * @param sampled	array of the size of the given data (number of 
   * 			instances); position will be set to true if in
   * 			result
   * @return		the new sub-sample
   */
  public Instances resample(Instances data, int seed, boolean[] sampled) {
    Instances     newData;
    int           i;
    int           index;
    Instance      instance;
    int           bagSize;
    Random        random;
    
    random = new Random(seed);
    
    // calculate bagsize
    bagSize = calcBagSize(data);

    newData = new Instances(data, bagSize);
    
    if (data.numInstances() > 0) {
      for (i = 0; i < bagSize; i++) {
        index = random.nextInt(data.numInstances());
        instance = data.instance(index);
        sampled[index] = true;
        instance.setWeight(1);
        newData.add(instance);
      }
    }

    return newData;
  }

  /**
   * creates a new instance of the classifier, sets the parameters and 
   * trains the classifier.
   * 
   * @param nextSeed	the next seed value for the classifier
   * @return		the trained classifier
   * @throws Exception	if something goes wrong
   */
  protected Classifier initClassifier(int nextSeed) throws Exception {
    RandomTree        tree;

    tree = new RandomTree();
    tree.setKValue(m_KValue);
    tree.setMinNum(getMinNum());
    tree.setSeed(nextSeed);
    tree.buildClassifier(m_TrainsetNew);

    return tree;
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
    
    return result;
  }
  
  /**
   * performs the actual building of the classifier
   * 
   * @throws Exception if building fails
   */
  @Override
  protected void buildClassifier() throws Exception {
    Classifier        tree;
    int               i;
    int               n;
    int               nextSeed;
    double[]          dist;
    Instances         bagData;
    boolean[]         inBag;
    double            outOfBagCount;
    double            errorSum;
    Instance          outOfBagInst;

    m_PureTrainNodes = 0;
    m_PureTestNodes  = 0;
    
    for (i = 0; i < getNumTrees(); i++) {
      // info
      if (getVerbose())
        System.out.print(".");

      // get next seed number
      nextSeed = m_Random.nextInt();

      // bagging?
      if (getUseBagging()) {
        // inBag-dataset/array
        inBag   = new boolean[m_TrainsetNew.numInstances()];
        bagData = resample(m_TrainsetNew, nextSeed, inBag);
        
        // build i.th tree
        tree = initClassifier(nextSeed);

        // determine and store distributions
        for (n = 0; n < m_TestsetNew.numInstances(); n++) {
          dist = tree.distributionForInstance(m_TestsetNew.instance(n));
          m_List.addDistribution(m_TestsetNew.instance(n), dist);
        }

        // determine out-of-bag-error
        outOfBagCount = 0;
        errorSum      = 0;

        for (n = 0; n < inBag.length; n++) {  
          if (!inBag[n]) {
            outOfBagInst = m_TrainsetNew.instance(n);
            outOfBagCount += outOfBagInst.weight();
            if (m_TrainsetNew.classAttribute().isNumeric()) {
              errorSum += outOfBagInst.weight() *
                StrictMath.abs(tree.classifyInstance(outOfBagInst)
                         - outOfBagInst.classValue());
            } 
            else {
              if (tree.classifyInstance(outOfBagInst) 
                    != outOfBagInst.classValue()) {
                errorSum += outOfBagInst.weight();
              }
            }
          }
        }

        m_OutOfBagError = errorSum / outOfBagCount;
      }
      else {
        // build i.th tree
        tree = initClassifier(nextSeed);

        // determine and store distributions
        for (n = 0; n < m_TestsetNew.numInstances(); n++) {
          dist = tree.distributionForInstance(m_TestsetNew.instance(n));
          m_List.addDistribution(m_TestsetNew.instance(n), dist);
        }
      }

      // get information about pure nodes
      try {
        if (tree instanceof AdditionalMeasureProducer) {
          m_PureTrainNodes += ((AdditionalMeasureProducer) tree).getMeasure(
                                  "measurePureTrainNodes");
          m_PureTestNodes  += ((AdditionalMeasureProducer) tree).getMeasure(
                                  "measurePureTestNodes");
        }
      }
      catch (Exception e) {
        e.printStackTrace();
      }

      tree = null;
    }
      
    if (getVerbose())
      System.out.println();
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
   * here initialization and building, possible iterations will happen
   * 
   * @throws Exception	if something goes wrong
   */
  @Override
  protected void build() throws Exception {
    AttributeStats        stats;
    int                   i;
    
    // determine number of features to be selected
    m_KValue = getNumFeatures();
    if (m_KValue < 1) 
      m_KValue = (int) Utils.log2(m_Trainset.numAttributes()) + 1;

    // determine class distribution
    m_ClassDistribution = new double[2];
    stats = m_Trainset.attributeStats(m_Trainset.classIndex());
    for (i = 0; i < 2; i++) {
      if (stats.totalCount > 0)
        m_ClassDistribution[i] = stats.nominalCounts[i] / stats.totalCount;
      else
        m_ClassDistribution[i] = 0;
    }

    // the number of instances added to the training set in each iteration
    m_InstancesPerIteration =   (double) m_Testset.numInstances() 
                              / getFolds();
    if (getDebug())
      System.out.println("InstancesPerIteration: " + m_InstancesPerIteration);

    // build list of sorted test instances
    m_List = new RankedList(m_Testset, m_ClassDistribution);

    // build classifier
    m_Random = new Random(getSeed());
    for (i = 0; i <= getFolds(); i++) {
      if (getVerbose()) {
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

    result = "";

    if (getUseBagging())
      result = "Out of bag error.......: " + m_OutOfBagError + "\n";

    return result;
  }
  
  /**
   * returns some information about the parameters
   * 
   * @return		information about the parameters
   */
  @Override
  protected String toStringParameters() {
    String        result;
    String        size;

    result =   "Trees..................: " + getNumTrees() + "\n"
             + "Features...............: " + m_KValue + "\n"
             + "Min Number of instances: " + getMinNum() + "\n"
             + "Folds..................: " + getFolds() + "\n"
             + "CutOff.................: " + getCutOff() + "\n"
             + "Bagging................: " + getUseBagging() + "\n";

    if (getUseBagging()) {
      if (getBagSize() == 0.0)
        size = "100%";
      else if (getBagSize() < 0)
        size = Integer.toString((int) StrictMath.ceil(-getBagSize() * 100)) + "%";
      else
        size = Integer.toString((int) StrictMath.round(getBagSize()));
      result += "Bag-Size...............: " + size + "\n";
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
    return "";
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
    CollectiveEvaluationUtils.runClassifier(new CollectiveForest(), args);
  }
}
