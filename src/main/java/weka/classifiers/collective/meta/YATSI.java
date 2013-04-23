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
 * YATSI.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.meta;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.collective.CollectiveRandomizableSingleClassifierEnhancer;
import weka.classifiers.evaluation.CollectiveEvaluationUtils;
import weka.classifiers.lazy.IBk;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.neighboursearch.KDTree;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 <!-- globalinfo-start -->
 * "Yet Another Two Stage Idea" - A collective Classifier that uses the given classifier to train on the training set and labeling the unlabeled data. Predictions are then done via nearest-neighbor-search and a majority over the k nearest neigbors (actually their weights) of the instance that is to be predicted (in case of ties the first label is chosen).<br/>
 * Missing values are replaced with the ReplaceMissingValues filter.<br/>
 * <br/>
 * For more information, see:<br/>
 * <br/>
 * Kurt Driessens, Peter Reutemann, Bernhard Pfahringer, Claire Leschi: Using weighted nearest neighbor to benefit from unlabeled data. In: Advances in Knowledge Discovery and Data Mining, 10th Pacific-Asia Conference, PAKDD 2006, 60-69, 2006.
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inproceedings{Driessens2006,
 *    author = {Kurt Driessens and Peter Reutemann and Bernhard Pfahringer and Claire Leschi},
 *    booktitle = {Advances in Knowledge Discovery and Data Mining, 10th Pacific-Asia Conference, PAKDD 2006},
 *    pages = {60-69},
 *    series = {LNCS},
 *    title = {Using weighted nearest neighbor to benefit from unlabeled data},
 *    volume = {3918},
 *    year = {2006}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -A &lt;spec&gt;
 *  The specifiction of the nearest neighbor search algorithm.
 *  (default KDTree)</pre>
 * 
 * <pre> -F &lt;num&gt;
 *  The weighting factor of the unlabeled instances.
 *  (default 1.0)</pre>
 * 
 * <pre> -no-weights
 *  Disables any weighting.
 * </pre>
 * 
 * <pre> -K &lt;num&gt;
 *  The number of neighbors to consider in the neighborhood.
 *  (default 10)</pre>
 * 
 * <pre> -X
 *  Uses IBk's internal crossvalidation to determine an optimal K
 *  based on the training data.
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
 * @author Kurt Driessens (kurtd at cs dot waikato dot ac dot nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $ 
 * @see ReplaceMissingValues
 */
public class YATSI
  extends CollectiveRandomizableSingleClassifierEnhancer
  implements TechnicalInformationHandler {

  /** for serialization */
  private static final long serialVersionUID = 1564425216021508454L;

  /** the nearest neighbor search algorithm */
  protected NearestNeighbourSearch m_NNSearch;

  /** the weighting factor */
  protected double m_WeightingFactor;

  /** disables any weighting if set to TRUE */
  protected boolean m_NoWeights;

  /** the (initial) K value */
  protected int m_KNN;

  /** the (determined and used) K value */
  protected int m_KNNDetermined;

  /** whether to use IBk's crossvalidation to determine K */
  protected boolean m_UseCV;

  /** handles the processing/sorting/etc of the data */
  protected YATSIInstances m_Data;

  /** whether the data (test instances) is already labeled */
  protected boolean m_DataIsLabeled;

  /** the labeled test data */
  protected Instances m_TestsetNew;

  /** counter, whether the neighborhood contains only one class label, if
   * a prediction is done via the getDistribution method
   * @see #getDistribution(Instance) */
  protected int m_ClearCutDistribution;
  
  /** a counter for weight-induced flips */
  protected int m_WeightFlips;

  /**
   * performs initialization of members
   */
  @Override
  protected void initializeMembers() {
    super.initializeMembers();

    m_Classifier           = new weka.classifiers.trees.J48();
    m_NNSearch             = new KDTree();
    m_WeightingFactor      = 1.0;
    m_NoWeights            = false;
    m_KNN                  = 10;
    m_KNNDetermined        = 10;
    m_UseCV                = false;
    m_Data                 = null;
    m_DataIsLabeled        = false;
    m_TestsetNew           = null;
    m_ClearCutDistribution = 0;
    m_WeightFlips          = 0;

    m_AdditionalMeasures.add("measureClearCutDistribution");
    m_AdditionalMeasures.add("measureWeightFlips");
  }

  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return    
        "\"Yet Another Two Stage Idea\" - A collective Classifier that uses "
      + "the given classifier to train on the training set and labeling the "
      + "unlabeled data. Predictions are then done via "
      + "nearest-neighbor-search and a majority over the k nearest "
      + "neigbors (actually their weights) of the instance that is to be "
      + "predicted (in case of ties the first label is chosen).\n"
      + "Missing values are replaced with the ReplaceMissingValues filter.\n\n"
      + "For more information, see:\n\n"
      + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    
    result = new TechnicalInformation(Type.INPROCEEDINGS);
    result.setValue(Field.AUTHOR, "Kurt Driessens and Peter Reutemann and Bernhard Pfahringer and Claire Leschi");
    result.setValue(Field.TITLE, "Using weighted nearest neighbor to benefit from unlabeled data");
    result.setValue(Field.BOOKTITLE, "Advances in Knowledge Discovery and Data Mining, 10th Pacific-Asia Conference, PAKDD 2006");
    result.setValue(Field.YEAR, "2006");
    result.setValue(Field.PAGES, "60-69");
    result.setValue(Field.SERIES, "LNCS");
    result.setValue(Field.VOLUME, "3918");
    
    return result;
  }
  
  /**
   * String describing default classifier.
   * 
   * @return 		the classname
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
        "\tThe specifiction of the nearest neighbor search algorithm.\n"
        + "\t(default KDTree)",
        "A", 1, "-A <spec>"));
    
    result.addElement(new Option(
        "\tThe weighting factor of the unlabeled instances.\n"
        + "\t(default 1.0)",
        "F", 1, "-F <num>"));
    
    result.addElement(new Option(
        "\tDisables any weighting.\n",
        "no-weights", 0, "-no-weights"));
    
    result.addElement(new Option(
        "\tThe number of neighbors to consider in the neighborhood.\n"
        + "\t(default 10)",
        "K", 1, "-K <num>"));
    
    result.addElement(new Option(
        "\tUses IBk's internal crossvalidation to determine an optimal K\n"
        + "\tbased on the training data.\n",
        "X", 0, "-X"));
    
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
   * <pre> -A &lt;spec&gt;
   *  The specifiction of the nearest neighbor search algorithm.
   *  (default KDTree)</pre>
   * 
   * <pre> -F &lt;num&gt;
   *  The weighting factor of the unlabeled instances.
   *  (default 1.0)</pre>
   * 
   * <pre> -no-weights
   *  Disables any weighting.
   * </pre>
   * 
   * <pre> -K &lt;num&gt;
   *  The number of neighbors to consider in the neighborhood.
   *  (default 10)</pre>
   * 
   * <pre> -X
   *  Uses IBk's internal crossvalidation to determine an optimal K
   *  based on the training data.
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

    tmpStr = Utils.getOption('A', options);
    if (tmpStr.length() != 0) {
      tmpOptions    = Utils.splitOptions(tmpStr);
      tmpStr        = tmpOptions[0];
      tmpOptions[0] = "";
      setNearestNeighbourSearchAlgorithm( 
          (NearestNeighbourSearch)
          Utils.forName( NearestNeighbourSearch.class, 
            tmpStr, 
            tmpOptions)
          );
    }
    else {
      setNearestNeighbourSearchAlgorithm(new KDTree());
    }
    
    tmpStr = Utils.getOption('F', options);
    if (tmpStr.length() != 0)
      setWeightingFactor(Double.parseDouble(tmpStr));
    else
      setWeightingFactor(1.0);
    
    setNoWeights(Utils.getFlag("no-weights", options));
    
    tmpStr = Utils.getOption('K', options);
    if (tmpStr.length() != 0)
      setKNN(Integer.parseInt(tmpStr));
    else
      setKNN(10);
    
    setUseCV(Utils.getFlag('X', options));
    
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

    result.add("-A");
    result.add(getSpecification(m_NNSearch));
    
    if (getNoWeights()) {
      result.add("-no-weights");
    }
    else {
      result.add("-F");
      result.add("" + getWeightingFactor());
    }
    
    result.add("-K");
    result.add("" + getKNN());
    
    if (getUseCV())
      result.add("-X");
    
    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);
    
    return (String[]) result.toArray(new String[result.size()]);
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String KNNTipText() {
    return "The number of neighbours to use.";
  }
  
  /**
   * Set the number of neighbours the learner is to use.
   *
   * @param value the number of neighbours.
   */
  public void setKNN(int value) {
    if (value > 0)
      m_KNN = value;
    else
      System.out.println(
          "K needs to be greater than 0 ( provided: " + value + ")!");
  }

  /**
   * Gets the number of neighbours the learner will use.
   *
   * @return the number of neighbours.
   */
  public int getKNN() {
    return m_KNN;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String weightingFactorTipText() {
    return "The weighting factor for the unlabeled instances.";
  }
  
  /**
   * Set the weighting factor for the unlabeled instances
   *
   * @param value the weighting factor.
   */
  public void setWeightingFactor(double value) {
    if (value > 0)
      m_WeightingFactor = value;
    else
      System.out.println(
          "Weighting factor needs to be greater than 0 ( provided: " 
          + value + ")!");
  }

  /**
   * Gets the weighting factor for the unlabeled instances.
   *
   * @return the weighting factor.
   */
  public double getWeightingFactor() {
    return m_WeightingFactor;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String noWeightsTipText() {
    return "Disables any weighting.";
  }
  
  /**
   * Disables any weighting if TRUE.
   *
   * @param value whether to disable weighting or not
   */
  public void setNoWeights(boolean value) {
    m_NoWeights = value;
  }

  /**
   * Gets whether no weighting is done at all.
   *
   * @return whether weighting is disabled.
   */
  public boolean getNoWeights() {
    return m_NoWeights;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   *         displaying in the explorer/experimenter gui
   */
  public String nearestNeighbourSearchAlgorithmTipText() {
    return "The nearest neighbour search algorithm to use (Default: KDTree).";
  }

  /**
   * Returns the current nearestNeighbourSearch algorithm in use.
   * @return the NearestNeighbourSearch algorithm currently in use.
   */
  public NearestNeighbourSearch getNearestNeighbourSearchAlgorithm() {
    return m_NNSearch;
  }

  /**
   * Sets the nearestNeighbourSearch algorithm to be used for finding nearest
   * neighbour(s).
   * @param value The NearestNeighbourSearch class.
   */
  public void setNearestNeighbourSearchAlgorithm(NearestNeighbourSearch value) {
    m_NNSearch = value;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String useCVTipText() {
    return "Whether to use IBk's internal crossvalidation to determine an optimal K value on the training data, which is then used as K.";
  }
  
  /**
   * Sets whether to use IBk's internal crossvalidation to determine an
   * optimal K value on the training data, which is then used as K.
   *
   * @param value whether to use IBk's internal CV.
   */
  public void setUseCV(boolean value) {
    m_UseCV = value;
  }

  /**
   * Gets whether IBk's internal crossvalidation is used to determine an
   * optimal K value on the training data, which is then used as K.
   *
   * @return whether to use IBk's internal CV.
   */
  public boolean getUseCV() {
    return m_UseCV;
  }

  /**
   * Returns the value of the named measure
   * @param measureName the name of the measure to query for its value
   * @return the value of the named measure
   * @throws IllegalArgumentException if the named measure is not supported
   */
  @Override
  public double getMeasure(String measureName) {
    if (measureName.equalsIgnoreCase("measureClearCutDistribution"))
      return m_ClearCutDistribution;
    else if (measureName.equalsIgnoreCase("measureWeightFlips"))
        return m_WeightFlips;
    else
      return super.getMeasure(measureName);
  }
  
  /**
   * resets the classifier
   */
  @Override
  public void reset() {
    super.reset();

    m_DataIsLabeled        = false;
    m_ClearCutDistribution = 0;
    m_WeightFlips          = 0;
  }

  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();

    // class
    result.disableAllClasses();
    result.disableAllClassDependencies();
    Capabilities cls = getClassifier().getCapabilities();
    if (cls.handles(Capability.NOMINAL_CLASS))
      result.enable(Capability.NOMINAL_CLASS);
    else if (cls.handles(Capability.BINARY_CLASS))
      result.enable(Capability.BINARY_CLASS);
    else if (cls.handles(Capability.UNARY_CLASS))
      result.enable(Capability.UNARY_CLASS);
    result.disable(Capability.MISSING_CLASS_VALUES);
    
    return result;
  }
  
  /**
   * builds the necessary CollectiveInstances from the given Instances
   * @throws Exception if anything goes wrong
   */
  @Override
  protected void generateSets() throws Exception {
    super.generateSets();

    if (!m_DataIsLabeled)
      m_TestsetNew = new Instances(m_Testset);
    else
      m_Data = new YATSIInstances(this, m_Trainset, m_TestsetNew, true);
  }
  
  /**
   * performs the actual building of the classifier (feeds the base classifier
   * with the training instances, that were filtered with ReplaceMissingValues)
   * @throws Exception if building fails
   */
  @Override
  protected void buildClassifier() throws Exception {
    ReplaceMissingValues      missing;
    
    missing = new ReplaceMissingValues();
    missing.setInputFormat(m_Trainset);

    m_Classifier.buildClassifier(Filter.useFilter(m_Trainset, missing));
  }
  
  /**
   * builds the base classifier and uses that one for labeling the unlabeled
   * test instances
   * 
   * @see		#buildClassifier()
   * @throws Exception	if something goes wrong
   */
  @Override
  protected void build() throws Exception {
    int           i;
    Instance      inst;
    IBk           ibk;

    // determine K?
    if (getUseCV()) {
      ibk = new IBk();
      ibk.setKNN(getKNN());
      ibk.setCrossValidate(true);
      ibk.setWindowSize(0);
      ibk.setMeanSquared(false);
      m_KNNDetermined = ibk.getKNN();
    }
    else {
      m_KNNDetermined = getKNN();
    }
    
    // build classifier
    buildClassifier();

    // label unlabeled instances
    for (i = 0; i < m_TestsetNew.numInstances(); i++) {
      inst = m_TestsetNew.instance(i);
      inst.setClassValue(m_Classifier.classifyInstance(inst));
    }
    m_DataIsLabeled = true;

    // generate combined dataset
    generateSets();

    // initialize nearest neighbor search algorithm
    m_NNSearch.setInstances(m_Data.getTrainSet());
  }
  
  /**
   * internal function for determining the class distribution for an instance, 
   * will be overridden by derived classes. <br/>
   * 
   * @param instance	the instance to get the distribution for
   * @return		the distribution for the given instance
   * @throws Exception	if something goes wrong
   */
  @Override
  protected double[] getDistribution(Instance instance) throws Exception {
    int         index;
    int         i;
    double[]    result;
    Instances   neighbors;
    Instance    inst;
    double[]    count;
    double[]    countNum;
    int         labelIndex;

    result = null;

    // find instance
    index = m_Data.indexOf(instance);
    if (index > -1) {
      // get neighbors
      neighbors = m_NNSearch.kNearestNeighbours(
                    m_Data.get(index), m_KNNDetermined);

      // count class label
      count    = new double[neighbors.numClasses()];
      countNum = new double[neighbors.numClasses()];
      for (i = 0; i < neighbors.numInstances(); i++) {
        inst = neighbors.instance(i);
        if (!inst.classIsMissing()) {
          count[(int) inst.classValue()] += inst.weight();
          countNum[(int) inst.classValue()]++;
        }
      }

      // build result
      result = new double[instance.numClasses()];
      for (i = 0; i < result.length; i++)
        result[i] = count[i];
      if (Utils.gr(Utils.sum(result), 0))
        Utils.normalize(result);
      else
        System.out.println(
            "No summed up weights: " + instance 
            + ", counts=" + Utils.arrayToString(countNum));
      labelIndex = Utils.maxIndex(count);
      // is it a clear-cut distribution?
      if (!Utils.eq(Utils.sum(count) - count[labelIndex], 0))
        m_ClearCutDistribution++;
      // did the label change due to weights?
      if (Utils.maxIndex(countNum) != labelIndex)
        m_WeightFlips++;
    }
    else {
      throw new Exception("Cannot find instance: " + instance + "\n" 
          + " -> pos=" + index 
          + " = " + m_Data.get(StrictMath.abs(index)));
    }

    return result;
  }
  
  /**
   * returns some information about the parameters
   * 
   * @return		information about the parameters
   */
  @Override
  protected String toStringParameters() {
    String     result;
    
    if (checkBuiltStatus()) {
      result  = "";
      result += "Used K................: " + m_KNNDetermined + "\n";
      result += "Use CV for K..........: " + getUseCV() + "\n";
      result += "Using weights.........: " + (!getNoWeights()) + "\n";
      result += "Weighting factor......: " + getWeightingFactor() + "\n";
      result += "NN search algorithm...: " + getSpecification(m_NNSearch) +"\n";
      result += "\n";
    }
    else {
      result = "";
      result += "Used K................: " + "-not yet determined-" + "\n";
      result += "Use CV for K..........: " + getUseCV() + "\n";
      result += "Using weights.........: " + (!getNoWeights()) + "\n";
      result += "Weighting factor......: " + getWeightingFactor() + "\n";
      result += "NN search algorithm...: " + getSpecification(m_NNSearch) +"\n";
      result += "\n";
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
    StringBuffer     text;
    
    text = new StringBuffer();
    text.append(super.toStringModel());
    text.append("\n");
    text.append("The built model for labeling the unlabeled data:\n");
    text.append("\n");
    text.append(m_Classifier.toString());
    
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
    CollectiveEvaluationUtils.runClassifier(new YATSI(), args);
  }
  
  /* ********************* other classes ************************** */
  
  /**
   * Stores the relation between unprocessed instance and processed instance.
   *
   * @author FracPete (fracpete at waikato dot ac dot nz)
   */
  protected class YATSIInstances 
    implements Serializable {

    /** for serialization */
    private static final long serialVersionUID = 1763148429358814942L;

    /** the parent algorithm (used to get the parameters) */
    protected YATSI m_Parent = null;
    
    /** the unprocessed instances */
    protected Instance[] m_Unprocessed = null;

    /** the weights of the instances */
    protected double[] m_Weights = null;

    /** the new training set */
    protected Instances m_Trainset = null;

    /** for finding instances again (used for classifying) */
    protected InstanceComparator m_Comparator = new InstanceComparator(false);

    /** The filter used to get rid of missing values. */
    protected ReplaceMissingValues m_Missing;

    /**
     * initializes the object
     * @param parent      the parent algorithm
     * @param train       the train instances
     * @param test        the test instances
     * @param setWeights  whether to set the weights for the training set 
     *                    (the processed instances)
     * @throws Exception  if something goes wrong
     */
    public YATSIInstances(YATSI parent, Instances train, Instances test, 
                          boolean setWeights) 
      throws Exception {

      super();

      m_Parent = parent;

      // build sorted array (train + test)
      double weight;
      if (getParent().getNoWeights())
        weight = 1.0;
      else
        weight =   (double) train.numInstances() 
                 / (double) test.numInstances()
                 * getParent().getWeightingFactor();
      m_Unprocessed = new Instance[train.numInstances() + test.numInstances()];
      for (int i = 0; i < train.numInstances(); i++)
        m_Unprocessed[i] = train.instance(i);
      for (int i = 0; i < test.numInstances(); i++) {
        m_Unprocessed[train.numInstances() + i] = test.instance(i);
        m_Unprocessed[train.numInstances() + i].setWeight(weight);
      }
      Arrays.sort(m_Unprocessed, m_Comparator);

      // weights
      m_Weights = new double[m_Unprocessed.length];
      for (int i = 0; i < m_Unprocessed.length; i++) {
        m_Weights[i] = m_Unprocessed[i].weight();
        if (!setWeights)
          m_Unprocessed[i].setWeight(1);
      }

      // filter data
      m_Trainset  = new Instances(train, 0);
      for (int i = 0; i < m_Unprocessed.length; i++)
        m_Trainset.add(m_Unprocessed[i]);

      // set up filter
      m_Missing = new ReplaceMissingValues();
      m_Missing.setInputFormat(m_Trainset);
      m_Trainset = Filter.useFilter(m_Trainset, m_Missing); 
    }

    /**
     * returns the parent algorithm
     * 
     * @return		the parent
     */
    public YATSI getParent() {
      return m_Parent;
    }

    /**
     * returns the train set (with the processed instances)
     * 
     * @return		the train set
     */
    public Instances getTrainSet() {
      return m_Trainset;
    }

    /**
     * returns the number of stored instances
     * 
     * @return		the number of instances
     */
    public int size() {
      return m_Trainset.numInstances();
    }

    /**
     * returns the index of the given (unprocessed) Instance, -1 in case it
     * can't find the instance
     * 
     * @param inst	the instance to get the index for
     * @return		the index of the instance, -1 if not found
     */
    public int indexOf(Instance inst) {
      return Arrays.binarySearch(m_Unprocessed, inst, m_Comparator);
    }

    /**
     * returns the processed instance for the given index, null if not within
     * bounds.
     * 
     * @param index	the index of the instance to retrieve
     * @return		the processed instance, or null if index out of bounds
     */
    public Instance get(int index) {
      if ( (index >= 0) && (index < m_Trainset.numInstances()) )
        return m_Trainset.instance(index);
      else
        return null;
    }

    /**
     * returns the processed version of the unprocessed instance in the new
     * training set, null if it can't find the instance
     * @param inst      the unprocessed instance to retrieve the processed one
     *                  for
     * @return          the processed version of the given instance 
     * @see             #getTrainSet()
     */
    public Instance get(Instance inst) {
      return get(indexOf(inst));
    }

    /**
     * returns the weight of the processed instance for the given index, -1 if
     * not within bounds.
     * 
     * @param index	the index of the processed instance
     * @return		the weight, -1 if index not in bounds
     */
    public double getWeight(int index) {
      if (get(index) != null)
        return m_Weights[index];
      else
        return -1;
    }

    /**
     * returns the weight of the processed version of the unprocessed instance
     * in the new training set, -1 if it can't find the instance
     * @param inst      the unprocessed instance to retrieve the processed one
     *                  for
     * @return          weight of the processed version of the given instance 
     */
    public double getWeight(Instance inst) {
      return getWeight(indexOf(inst));
    }
  }
}
