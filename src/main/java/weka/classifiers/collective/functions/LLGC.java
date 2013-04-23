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
 * LLGC.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.functions;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.collective.CollectiveRandomizableClassifier;
import weka.classifiers.evaluation.CollectiveEvaluationUtils;
import weka.classifiers.functions.SMO;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.matrix.Matrix;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;

/**
 <!-- globalinfo-start -->
 * A collective classifier that generates a smooth classifying function for labeled and unlabeled data. It pre-processes the data with the following filters:<br/>
 *  - weka.filters.unsupervised.attribute.ReplaceMissingValues<br/>
 *  - weka.filters.unsupervised.attribute.NominalToBinary<br/>
 *  - if -N 0 then weka.filters.unsupervised.attribute.Normalize<br/>
 *  - if -N 1 then weka.filters.unsupervised.attribute.Standardize<br/>
 * <br/>
 * For more informations, refer to the following paper:<br/>
 * <br/>
 * Dengyong Zhou, Olivier Bousquet, Thomas Navin Lal, Jason Weston, Bernhard Schoelkopf}: Learning with Local and Global Consistency. In: Advances in Neural Information Processing Systems 16, , 2003.<br/>
 * <br/>
 * The following modification was done (option -include-atts):<br/>
 * - the distance between two instances is not divided by 2*sigma^2, but by 2*sigm^2*N, with N as the number ofattributes.
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inproceedings{Zhou2003,
 *    author = {Dengyong Zhou and Olivier Bousquet and Thomas Navin Lal and Jason Weston and Bernhard Schoelkopf}},
 *    booktitle = {Advances in Neural Information Processing Systems 16},
 *    publisher = {MIT Press},
 *    title = {Learning with Local and Global Consistency},
 *    year = {2003},
 *    PDF = {http://books.nips.cc/papers/files/nips16/NIPS2003_AA41.pdf}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
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
 * <pre> -alpha &lt;num&gt;
 *  The iteration parameter alpha.
 *  (default 0.99)</pre>
 * 
 * <pre> -sigma &lt;num&gt;
 *  The sigma used in the affinity matrix.
 *  (default 1.0)</pre>
 * 
 * <pre> -limit &lt;num&gt;
 *  The sequence limit function to use.
 * </pre>
 * 
 * <pre> -repeats &lt;num&gt;
 *  The number of times to repeat the iteration after
 *  convergence. (default 0)
 * </pre>
 * 
 * <pre> -distance &lt;spec&gt;
 *  The distance function to use.
 *  (default weka.core.EuclideanDistance)
 * </pre>
 * 
 * <pre> -include-atts
 *  Whether to include the size of the dataset in normalizing 
 *  the distance between instances for the affinity matrix.
 *  (default yes)
 * </pre>
 * 
 <!-- options-end -->
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $ 
 */
public class LLGC
  extends CollectiveRandomizableClassifier
  implements TechnicalInformationHandler {
  
  /** for serialization */
  private static final long serialVersionUID = -273113869266661708L;

  /** copy of the original training dataset */
  protected Instances m_TrainsetNew;
  
  /** copy of the original test dataset */
  protected Instances m_TestsetNew;
  
  /** the alpha used in the iterations */
  protected double m_Alpha;

  /** the sigma used for the affinity matrix */
  protected double m_Sigma;

  /** the number of times to repeat the iterations after convergence (= p-1) */
  protected int m_Repeats;

  /** instance weighting method: graph kernel */
  public static final int SEQ_LIMIT_GRAPHKERNEL = 0;
  /** instance weighting method: stoachastic matrix */
  public static final int SEQ_LIMIT_STOCHASTICMATRIX = 1;
  /** instance weighting method: transposed stoachastic matrix */
  public static final int SEQ_LIMIT_STOCHASTICMATRIX_T = 2;
  /** instance weighting methods */
  public static final Tag[] TAGS_SEQ_LIMIT = {
    new Tag(SEQ_LIMIT_GRAPHKERNEL, "Graph Kernel: F'=((I-alpha*S)^-1)*Y"),
    new Tag(SEQ_LIMIT_STOCHASTICMATRIX, "Stoch. Matrix: F'=((I-alpha*P)^-1)*Y"),
    new Tag(SEQ_LIMIT_STOCHASTICMATRIX_T, "trans. Stoch. Matrix: F'=((I-alpha*P^T)^-1)*Y")
  };

  /** the sequence limit method to use */
  protected int m_SequenceLimit;

  /** Whether to normalize/standardize/neither */
  protected int m_filterType;

  /** Whether to include the number of attributes in normalizing the distance */
  protected boolean m_IncludeNumAttributes;

  /** the matrix for the class labels */
  protected Matrix m_MatrixY;

  /** the affinity matrix */
  protected Matrix m_MatrixW;

  /** the diagonal matrix */
  protected Matrix m_MatrixD;

  /** the spread matrix */
  protected Matrix m_MatrixS;

  /** the convergence matrix */
  protected Matrix m_MatrixFStar;

  /** handles the processing/sorting/etc of the data */
  protected LLGCInstances m_Data;

  /** the distance function to use */
  protected DistanceFunction m_DistanceFunction;

  /**
   * performs initialization of members
   */
  @Override
  protected void initializeMembers() {
    super.initializeMembers();
    
    m_TrainsetNew          = null;
    m_TestsetNew           = null;
    m_Alpha                = 0.99;
    m_Sigma                = 1.0;
    m_Repeats              = 0;
    m_SequenceLimit        = SEQ_LIMIT_GRAPHKERNEL;
    m_filterType           = SMO.FILTER_NORMALIZE;
    m_IncludeNumAttributes = true;
    m_MatrixY              = null;
    m_MatrixW              = null;
    m_MatrixD              = null;
    m_MatrixS              = null;
    m_MatrixFStar          = null;
    m_Data                 = null;
    m_DistanceFunction     = new EuclideanDistance();
  }

  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return    
        "A collective classifier that generates a smooth classifying function "
      + "for labeled and unlabeled data. It pre-processes the data with the "
      + "following filters:\n"
      + " - weka.filters.unsupervised.attribute.ReplaceMissingValues\n"
      + " - weka.filters.unsupervised.attribute.NominalToBinary\n"
      + " - if -N 0 then weka.filters.unsupervised.attribute.Normalize\n"
      + " - if -N 1 then weka.filters.unsupervised.attribute.Standardize\n"
      + "\n"
      + "For more informations, refer to the following paper:\n\n"
      + getTechnicalInformation().toString() + "\n\n"
      + "The following modification was done (option -include-atts):\n"
      + "- the distance between two instances is not divided by "
      + "2*sigma^2, but by 2*sigm^2*N, with N as the number of"
      + "attributes.";
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
    result.setValue(Field.AUTHOR, "Dengyong Zhou and Olivier Bousquet and Thomas Navin Lal and Jason Weston and Bernhard Schoelkopf}");
    result.setValue(Field.TITLE, "Learning with Local and Global Consistency");
    result.setValue(Field.BOOKTITLE, "Advances in Neural Information Processing Systems 16");
    result.setValue(Field.YEAR, "2003");
    result.setValue(Field.PUBLISHER, "MIT Press");
    result.setValue(Field.PDF, "http://books.nips.cc/papers/files/nips16/NIPS2003_AA41.pdf");
    
    return result;
  }
  
  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration listOptions() {
    Vector        result;
    Enumeration   en;
    
    result = new Vector();
    
    en = super.listOptions();
    while (en.hasMoreElements())
      result.addElement(en.nextElement());
    
    result.addElement(new Option(
        "\tThe iteration parameter alpha.\n"
        + "\t(default 0.99)",
        "alpha", 1, "-alpha <num>"));
    
    result.addElement(new Option(
        "\tThe sigma used in the affinity matrix.\n"
        + "\t(default 1.0)",
        "sigma", 1, "-sigma <num>"));
    
    result.addElement(new Option(
        "\tThe sequence limit function to use.\n",
        "limit", 1, "-limit <num>"));
    
    result.addElement(new Option(
        "\tThe number of times to repeat the iteration after\n"
        + "\tconvergence. (default 0)\n",
        "repeats", 1, "-repeats <num>"));
    
    result.addElement(new Option(
        "\tThe distance function to use.\n"
        + "\t(default weka.core.EuclideanDistance)\n",
        "distance", 1, "-distance <spec>"));
    
    result.addElement(new Option(
        "\tWhether to include the size of the dataset in normalizing \n"
        + "\tthe distance between instances for the affinity matrix.\n"
        + "\t(default yes)\n",
        "include-atts", 0, "-include-atts"));
    
    return result.elements();
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
   * <pre> -alpha &lt;num&gt;
   *  The iteration parameter alpha.
   *  (default 0.99)</pre>
   * 
   * <pre> -sigma &lt;num&gt;
   *  The sigma used in the affinity matrix.
   *  (default 1.0)</pre>
   * 
   * <pre> -limit &lt;num&gt;
   *  The sequence limit function to use.
   * </pre>
   * 
   * <pre> -repeats &lt;num&gt;
   *  The number of times to repeat the iteration after
   *  convergence. (default 0)
   * </pre>
   * 
   * <pre> -distance &lt;spec&gt;
   *  The distance function to use.
   *  (default weka.core.EuclideanDistance)
   * </pre>
   * 
   * <pre> -include-atts
   *  Whether to include the size of the dataset in normalizing 
   *  the distance between instances for the affinity matrix.
   *  (default yes)
   * </pre>
   * 
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {
    String      tmpStr;
    String      classname;
    String[]    spec;
    
    super.setOptions(options);

    tmpStr = Utils.getOption("alpha", options);
    if (tmpStr.length() != 0)
      setAlpha(Double.parseDouble(tmpStr));
    else
      setAlpha(0.99);

    tmpStr = Utils.getOption("sigma", options);
    if (tmpStr.length() != 0)
      setSigma(Double.parseDouble(tmpStr));
    else
      setSigma(1.0);

    tmpStr = Utils.getOption("limit", options);
    if (tmpStr.length() != 0)
      setSequenceLimit(
          new SelectedTag(Integer.parseInt(tmpStr), TAGS_SEQ_LIMIT));
    else
      setSequenceLimit(
          new SelectedTag(SEQ_LIMIT_GRAPHKERNEL, TAGS_SEQ_LIMIT));

    tmpStr = Utils.getOption("repeats", options);
    if (tmpStr.length() != 0)
      setRepeats(Integer.parseInt(tmpStr));
    else
      setRepeats(0);

    tmpStr = Utils.getOption('N', options);
    if (tmpStr.length() != 0)
      setFilterType(
          new SelectedTag(Integer.parseInt(tmpStr), SMO.TAGS_FILTER));
    else
      setFilterType(
          new SelectedTag(SMO.FILTER_NORMALIZE, SMO.TAGS_FILTER));

    tmpStr = Utils.getOption("distance", options);
    if (tmpStr.length() != 0) {
      spec = Utils.splitOptions(tmpStr);
      if (spec.length == 0)
        throw new Exception("Invalid DistanceFunction specification string."); 
      classname = spec[0];
      spec[0]   = "";

      setDistanceFunction((DistanceFunction)
          Utils.forName(DistanceFunction.class, classname, spec));
    }
    else {
      setDistanceFunction(new EuclideanDistance());  
    }

    setIncludeNumAttributes(Utils.getFlag("include-atts", options));
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
    
    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);

    result.add("-alpha");
    result.add("" + getAlpha());

    result.add("-sigma");
    result.add("" + getSigma());

    result.add("-limit");
    result.add("" + m_SequenceLimit);
    
    result.add("-repeats");
    result.add("" + m_Repeats);
    
    result.add("-N");
    result.add("" + m_filterType);
    
    result.add("-distance");
    result.add(getSpecification(getDistanceFunction()));

    if (getIncludeNumAttributes())
      result.add("-include-atts");
    
    return (String[]) result.toArray(new String[result.size()]);
  }
  
  /**
   * Sets the alpha parameter for the iterations.
   * 
   * @param value	the alpha to use
   */
  public void setAlpha(double value) {
    if ( (value >= 0) && (value <= 1) )
      m_Alpha = value;
    else
      System.out.println("0 <= alpha <= 1 (provided: " + value + ")!");
  }
  
  /**
   * Gets the alpha parameter for the iterations.
   *
   * @return the alpha parameter for the iterations
   */
  public double getAlpha() {
    return m_Alpha;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String alphaTipText() {
    return "The alpha parameter for the iterations.";
  }
  
  /**
   * Sets the sigma parameter for the calculation of the affinity matrix.
   * 
   * @param value	the sigma parameter
   */
  public void setSigma(double value) {
    if (value > 0)
      m_Sigma = value;
    else
      System.out.println("sigma > 0 (provided: " + value + ")!");
  }
  
  /**
   * Gets the sigma parameter for the calculation of the affinity matrix.
   *
   * @return the sigma parameter for the calculation of the affinity matrix.
   */
  public double getSigma() {
    return m_Sigma;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String sigmaTipText() {
    return "The sigma parameter for the calculation of the affinity matrix.";
  }
  
  /**
   * Sets the number of repeats of iteration after convergence (= p-1).
   * 
   * @param value	the number of repeats
   */
  public void setRepeats(int value) {
    if (value >= 0)
      m_Repeats = value;
    else
      System.out.println("repeats >= 0 (provided: " + value + ")!");
  }
  
  /**
   * Gets the number of repeats of iterations after convergence (= p-1).
   *
   * @return the number of repeats
   */
  public int getRepeats() {
    return m_Repeats;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String repeatsTipText() {
    return "The number of repeats of iterations after convergence (= p-1).";
  }
  
  /**
   * Gets the sequence limit function.
   *
   * @return the sequence limit function.
   * @see #TAGS_SEQ_LIMIT
   */
  public SelectedTag getSequenceLimit() {
    return new SelectedTag(m_SequenceLimit, TAGS_SEQ_LIMIT);
  }
  
  /**
   * Sets the sequence limit function.
   *
   * @param value the new sequence limit function.
   * @see #TAGS_SEQ_LIMIT
   */
  public void setSequenceLimit(SelectedTag value) {
    if (value.getTags() == TAGS_SEQ_LIMIT)
      m_SequenceLimit = value.getSelectedTag().getID();
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String sequenceLimitTipText() {
    return "The sequence limit function to use.";
  }
  
  /**
   * Gets how the training data will be transformed. Will be one of
   * SMO.FILTER_NORMALIZE, SMO.FILTER_STANDARDIZE, SMO.FILTER_NONE.
   *
   * @return the filtering mode
   */
  public SelectedTag getFilterType() {
    return new SelectedTag(m_filterType, SMO.TAGS_FILTER);
  }
  
  /**
   * Sets how the training data will be transformed. Should be one of
   * SMO.FILTER_NORMALIZE, SMO.FILTER_STANDARDIZE, SMO.FILTER_NONE.
   *
   * @param value the new filtering mode
   */
  public void setFilterType(SelectedTag value) {
    if (value.getTags() == SMO.TAGS_FILTER)
      m_filterType = value.getSelectedTag().getID();
  }

  /**
   * Returns the tip text for this property
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String filterTypeTipText() {
    return "Determines how/if the data will be transformed.";
  }
  
  /**  
   * Returns the tip text for this property
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String distanceFunctionTipText() {
    return   "The distance function to use for finding neighbours "
           + "(default: weka.core.EuclideanDistance). ";
  }

  /** 
   * returns the distance function currently in use 
   * 
   * @return		the current distance function
   */
  public DistanceFunction getDistanceFunction() {
    return m_DistanceFunction;
  }

  /** 
   * sets the distance function to use 
   * 
   * @param value	the distance function to use
   */
  public void setDistanceFunction(DistanceFunction value) {
    m_DistanceFunction = value;
  }
  
  /**  
   * Returns the tip text for this property
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String includeNumAttributesTipText() {
    return   "Whether to include the number of attributes in the calculation "
           + "of the affinity matrix (default yes).";
  }

  /** 
   * returns whether the number of attributes is used in the calculation of the
   * affinity matrix
   * 
   * @return		true if the number of attribute is also included
   */
  public boolean getIncludeNumAttributes() {
    return m_IncludeNumAttributes;
  }

  /** 
   * sets whether to include the number of attributes in the calculation of the
   * affinity matrix
   * 
   * @param value	if true the number of attributes are also considered
   */
  public void setIncludeNumAttributes(boolean value) {
    m_IncludeNumAttributes = value;
  }

  /**
   * resets the classifier
   */
  @Override
  public void reset() {
    super.reset();
    
    // clean up memory
    m_MatrixY = null;
    m_MatrixW = null;
    m_MatrixD = null;
    m_MatrixS = null;
    m_MatrixFStar = null;
    System.gc();
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
    result.enable(Capability.NOMINAL_CLASS);
    
    return result;
  }

  /**
   * generates copies of the original datasets and also builds a relation
   * (hashtable) between each original instance and new instance. This is
   * necessary to retrieve the determined class value in the 
   * <code>getDistribution(Instance)</code> method.
   * 
   * @see   #getDistribution(Instance)
   * @throws Exception if anything goes wrong
   */
  @Override
  protected void generateSets() throws Exception {
    super.generateSets();

    m_Data        = new LLGCInstances(this, m_Trainset, m_Testset);
    m_TrainsetNew = m_Data.getTrainSet();
    m_TestsetNew  = null;
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
    int       index;
    int       i;
    double[]  result;
    double[]  row;

    result = null;

    // find instance
    index = m_Data.indexOf(instance);
    if (index > -1) {
      result = new double[m_MatrixFStar.getColumnDimension()];
      row    = new double[m_MatrixFStar.getColumnDimension()];
      for (i = 0; i < row.length; i++)
        row[i] = m_MatrixFStar.get(index, i);
      result[Utils.maxIndex(row)] = 1.0;
    }
    else {
      throw new Exception("Cannot find instance: " + instance + "\n" 
          + " -> pos=" + index 
          + " = " + m_Data.get(StrictMath.abs(index)));
    }

    return result;
  }

  /**
   * initializes the matrices
   * 
   * @throws Exception	if something goes wrong
   */
  protected void initialize() throws Exception {
    int           numInst;
    int           numCls;
    int           i;
    int           n;
    double        d;
    double        sum;
    double        factor;
    
    numInst = m_Data.size();
    numCls  = m_TrainsetNew.numClasses();
    
    // the classification matrix Y
    clock();
    m_MatrixY = new Matrix(numInst, numCls);
    for (i = 0; i < numInst; i++) {
      if (!m_Data.get(i).classIsMissing())
        m_MatrixY.set(i, (int) m_Data.get(i).classValue(), 1.0);
    }
    clock("Matrix Y");

    // the affinity matrix W
    // calc distances and variance of distances (i.e., sample variance)
    clock();
    if (getIncludeNumAttributes())
      factor = m_TrainsetNew.numAttributes();
    else
      factor = 1;
    m_DistanceFunction.setInstances(m_TrainsetNew);
    m_MatrixW = new Matrix(numInst, numInst);
    for (i = 0; i < numInst; i++) {
      for (n = 0; n < numInst; n++) {
        if (i == n) {
          d = 0;
        }
        else {
          d = m_DistanceFunction.distance(m_Data.get(i), m_Data.get(n));
          d = StrictMath.exp(
                -StrictMath.pow(d, 2) 
                / (2 * getSigma() * getSigma() * factor));
        }
        
        m_MatrixW.set(i, n, d);
      }
    }
    clock("Matrix W");
    
    // the diagonal matrix D
    clock();
    m_MatrixD = new Matrix(numInst, numInst);
    for (i = 0; i < numInst; i++) {
      sum = 0;
      for (n = 0; n < numInst; n++)
        sum += m_MatrixW.get(i, n);
      m_MatrixD.set(i, i, sum);
    }
    clock("Matrix D");

    // calc S or P (both results are stored in S for simplicity)
    clock();
    switch (m_SequenceLimit) {
      case SEQ_LIMIT_GRAPHKERNEL:
        // D^-1/2
        m_MatrixD = m_MatrixD.sqrt().inverse();
        // S = D^-1/2 * W * D^-1/2
        m_MatrixS = m_MatrixD.times(m_MatrixW).times(m_MatrixD);
        break;

      case SEQ_LIMIT_STOCHASTICMATRIX:
        // P = D^-1 * W
        m_MatrixS = m_MatrixD.inverse().times(m_MatrixW);
        break;
        
      case SEQ_LIMIT_STOCHASTICMATRIX_T:
        // P^T = (D^-1 * W)^T
        m_MatrixS = m_MatrixD.inverse().times(m_MatrixW).transpose();
        break;
        
      default: 
        throw new Exception("Unknown sequence limit function: " 
            + m_SequenceLimit + "!");
    }
    clock("Matrix S/P");
  }
  
  /**
   * performs the actual building of the classifier
   * @throws Exception if building fails
   */
  @Override
  protected void buildClassifier() throws Exception {
    Matrix      m;
    int         i;
    
    // calc: F(*) = (I - alpha*S)^-p * Y
    //       with S either S, P or P^T -> see initialize()
    
    // I
    clock();
    m_MatrixFStar = Matrix.identity(
        m_MatrixS.getRowDimension(), 
        m_MatrixS.getColumnDimension());
    clock("Matrix I");
    
    // - alpha*S
    clock();
    m_MatrixFStar.minusEquals(m_MatrixS.times(getAlpha()));
    clock("- alpha*S");

    // repeat after convergence? (I - alpha*S)^p
    if (getRepeats() > 0) {
      clock();
      m = m_MatrixFStar.copy();
      for (i = 0; i < getRepeats(); i++)
        m_MatrixFStar = m_MatrixFStar.times(m);
      clock("p Repeats");
    }

    // ^-1 * Y
    clock();
    m_MatrixFStar = m_MatrixFStar.inverse().times(m_MatrixY);
    clock("^-1 * Y");

    // clean up
    m         = null;
    m_MatrixY = null;
    m_MatrixW = null;
    m_MatrixD = null;
    m_MatrixS = null;
    System.gc();
  }

  /**
   * initializes the classifier and then calls <code>buildClassifier()</code>
   * 
   * @see		#initialize()
   * @see		#buildClassifier()
   * @throws Exception	if something goes wrong
   */
  @Override
  protected void build() throws Exception {
    initialize();
    buildClassifier();
  }
  
  /**
   * returns some information about the parameters
   * 
   * @return		information about the parameters
   */
  @Override
  protected String toStringParameters() {
    String     result;
    
    result  = "";
    result += "Alpha.................: " + getAlpha() + "\n";
    result += "Sigma.................: " + getSigma() + "\n";
    result += "Sequence Limit Func...: " + m_SequenceLimit 
            + " (" + getSequenceLimit().getSelectedTag().getReadable() + ")\n";
    result += "Repeats after Converg.: " + getRepeats() + "\n";
    result += "Include # of atts.....: " + getIncludeNumAttributes() + "\n";
    result += "Normalization type....: " + m_filterType + "\n";
    
    return result;
  }
  
  /**
   * returns nothing for this classifier.
   * 
   * @return		returns an empty string
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
    CollectiveEvaluationUtils.runClassifier(new LLGC(), args);
  }

  
  /* ********************* other classes ************************** */
  
  
  /**
   * Stores the relation between unprocessed instance and processed instance.
   *
   * @author FracPete (fracpete at waikato dot ac dot nz)
   */
  protected class LLGCInstances 
    implements Serializable {

    /** for serialization */
    private static final long serialVersionUID = 1975979462375468594L;

    /** the parent algorithm (used to get the parameters) */
    protected LLGC m_Parent = null;
    
    /** the unprocessed instances */
    protected Instance[] m_Unprocessed = null;

    /** the new training set */
    protected Instances m_Trainset = null;

    /** for finding instances again (used for classifying) */
    protected InstanceComparator m_Comparator = new InstanceComparator(false);

    /** The filter used to make attributes numeric. */
    protected NominalToBinary m_NominalToBinary;

    /** The filter used to standardize/normalize all values. */
    protected Filter m_Filter = null;

    /** The filter used to get rid of missing values. */
    protected ReplaceMissingValues m_Missing;

    /**
     * initializes the object
     * 
     * @param parent      the parent algorithm
     * @param train       the train instances
     * @param test        the test instances
     * @throws Exception  if something goes wrong
     */
    public LLGCInstances(LLGC parent, Instances train, Instances test) 
      throws Exception {

      super();

      m_Parent = parent;

      // set up filters
      m_Missing = new ReplaceMissingValues();
      m_Missing.setInputFormat(train);

      m_NominalToBinary = new NominalToBinary();
      m_NominalToBinary.setInputFormat(train);

      if (getParent().getFilterType().getSelectedTag().getID() 
          == SMO.FILTER_STANDARDIZE) {
        m_Filter = new Standardize();
        m_Filter.setInputFormat(train);
      } 
      else if (getParent().getFilterType().getSelectedTag().getID() 
          == SMO.FILTER_NORMALIZE) {
        m_Filter = new Normalize();
        m_Filter.setInputFormat(train);
      } 
      else {
        m_Filter = null;
      }

      // build sorted array (train + test)
      m_Unprocessed = new Instance[train.numInstances() + test.numInstances()];
      for (int i = 0; i < train.numInstances(); i++)
        m_Unprocessed[i] = train.instance(i);
      for (int i = 0; i < test.numInstances(); i++)
        m_Unprocessed[train.numInstances() + i] = test.instance(i);
      Arrays.sort(m_Unprocessed, m_Comparator);

      // filter data
      m_Trainset  = new Instances(train, 0);
      for (int i = 0; i < m_Unprocessed.length; i++)
        m_Trainset.add(m_Unprocessed[i]);

      m_Missing.setInputFormat(m_Trainset);
      m_Trainset = Filter.useFilter(m_Trainset, m_Missing); 

      m_NominalToBinary.setInputFormat(m_Trainset);
      m_Trainset = Filter.useFilter(m_Trainset, m_NominalToBinary);

      if (m_Filter != null) {
        m_Filter.setInputFormat(m_Trainset);
        m_Trainset = Filter.useFilter(m_Trainset, m_Filter); 
      }
    }

    /**
     * returns the parent algorithm
     * 
     * @return		the parent
     */
    public LLGC getParent() {
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
     * @param inst	the instance to return the index for
     * @return		the index for the instance, -1 if not found
     */
    public int indexOf(Instance inst) {
      return Arrays.binarySearch(m_Unprocessed, inst, m_Comparator);
    }

    /**
     * returns the processed instance for the given index, null if not within
     * bounds.
     * 
     * @param index	the index of the instance to retrieve
     * @return		null if index out of bounds, otherwise the instance
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
  }
}
