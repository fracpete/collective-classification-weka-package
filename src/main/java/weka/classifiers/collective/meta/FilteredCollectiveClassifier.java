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
 * FilteredCollectiveClassifier.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.meta;

import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.collective.CollectiveClassifier;
import weka.classifiers.collective.CollectiveRandomizableSingleClassifierEnhancer;
import weka.classifiers.evaluation.CollectiveEvaluationUtils;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 <!-- globalinfo-start -->
 * A meta classifier that takes a filter and a collective classifier as input.<br/>
 * The filter is only trained on the provided training set, but still applied to instances from the training and test set, as well as to any instance that gets passed to the meta classifier.
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -F &lt;spec&gt;
 *  The specifiction of the filter (classname + options).
 *  (default ReplaceMissingValues)</pre>
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
 *  (default: weka.classifiers.collective.meta.YATSI)</pre>
 * 
 * <pre> 
 * Options specific to classifier weka.classifiers.collective.meta.YATSI:
 * </pre>
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
 * @author Bernhard Pfahringer (bernhard at cs dot waikato dot ac dot nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $ 
 */
public class FilteredCollectiveClassifier
  extends CollectiveRandomizableSingleClassifierEnhancer {

  /** for serialization */
  private static final long serialVersionUID = 4564124294420896247L;

  /** the filter to use */
  protected Filter m_Filter;

  /** the filtered training data */
  protected Instances m_TrainsetNew;

  /** the filtered test data */
  protected Instances m_TestsetNew;

  /**
   * performs initialization of members
   */
  @Override
  protected void initializeMembers() {
    super.initializeMembers();
    
    m_Filter      = new ReplaceMissingValues();
    m_Classifier  = new YATSI();
    m_TrainsetNew = null;
    m_TestsetNew  = null;
  }

  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return    
        "A meta classifier that takes a filter and a collective classifier "
      + "as input.\n"
      + "The filter is only trained on the provided training set, but still "
      + "applied to instances from the training and test set, as well as to "
      + "any instance that gets passed to the meta classifier.";
  }
  
  /**
   * String describing default classifier.
   * 
   * @return		the classname
   */
  @Override
  protected String defaultClassifierString() {
    return YATSI.class.getName();
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
        "\tThe specifiction of the filter (classname + options).\n"
        + "\t(default ReplaceMissingValues)",
        "F", 1, "-F <spec>"));
    
    Enumeration en = super.listOptions();
    while (en.hasMoreElements())
      result.addElement(en.nextElement());
      
    return result.elements();
  }
  
  /**
   * Parses a given list of options. Valid options are: <p/>
   *
   * -D <br/>
   * Turn on debugging output.<p/>
   *
   * -W classname <br/>
   * Specify the full class name of a classifier as the basis for
   * collective classifying (required).<p/>
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
   * -S seed <br/>
   * Random number seed for resampling (default 1). <p/>
   *
   * -verbose <br/>
   * whether to output some more information during improving the classifier.
   * <p/>
   *
   * -insight <br/> 
   *  whether to use the labels of the original test set to output more
   *  statistics. <p/>
   *
   * -F class-spec <br/>
   * The classname and parameters for the filter 
   * (default ReplaceMissingValues). <p/>
   *
   * Options after -- are passed to the designated classifier.<p/>
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {
    String        tmpStr;
    String[]      tmpOptions;

    tmpStr = Utils.getOption('F', options);
    if (tmpStr.length() != 0) {
      tmpOptions    = Utils.splitOptions(tmpStr);
      tmpStr        = tmpOptions[0];
      tmpOptions[0] = "";
      setFilter( 
          (Filter) Utils.forName(Filter.class, tmpStr, tmpOptions));
    }
    else {
      setFilter(new ReplaceMissingValues());
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

    result.add("-F");
    result.add(getSpecification(m_Filter));
    
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
    Capabilities	result;
    
    if (getFilter() == null)
      result = super.getCapabilities();
    else
      result = getFilter().getCapabilities();
    
    // set dependencies
    for (Capability cap: Capability.values())
      result.enableDependency(cap);
    
    return result;
  }

  /**
   * Set the base learner. Must be a collectice Classifier!
   *
   * @param value the classifier to use.
   */
  @Override
  public void setClassifier(Classifier value) {
    if (!(value instanceof CollectiveClassifier))
      throw new IllegalArgumentException("Classifier must be a collective one!");
    else
      super.setClassifier(value);
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String filterTipText() {
    return "The filter to be used.";
  }

  /**
   * Sets the filter.
   *
   * @param value the filter with all options set.
   */
  public void setFilter(Filter value) {
    m_Filter = value;
  }

  /**
   * Gets the filter used.
   *
   * @return the filter
   */
  public Filter getFilter() {
    return m_Filter;
  }
  
  /**
   * resets the classifier
   */
  @Override
  public void reset() {
    super.reset();

    if (getClassifier() != null)
      ((CollectiveClassifier) getClassifier()).reset();
  }
  
  /**
   * builds the necessary CollectiveInstances from the given Instances
   * @throws Exception if anything goes wrong
   */
  @Override
  protected void generateSets() throws Exception {
    super.generateSets();

    // setup filter
    m_Filter.setInputFormat(m_Trainset);

    // filter datasets
    m_TrainsetNew = Filter.useFilter(m_Trainset, m_Filter);
    m_TestsetNew  = Filter.useFilter(m_Testset,  m_Filter);
  }
  
  /**
   * performs the actual building of the classifier (feeds the base classifier
   * with the training instances, that were filtered with the given filter)
   * @throws Exception if building fails
   */
  @Override
  protected void buildClassifier() throws Exception {
    ((CollectiveClassifier) m_Classifier).buildClassifier(
        m_TrainsetNew, m_TestsetNew);
  }
  
  /**
   * builds the base classifier and uses that one for labeling the unlabeled
   * test instances. Just calls buildClassifier().
   * 
   * @throws Exception	if something goes wrong
   * @see   		#buildClassifier()
   */
  @Override
  protected void build() throws Exception {
    buildClassifier();
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
    if (m_Filter.numPendingOutput() > 0)
      throw new Exception("Filter output queue not empty!");

    if (!m_Filter.input(instance))
      throw new Exception("Filter didn't make the test instance"
			  + " immediately available!");

    m_Filter.batchFinished();
    Instance newInstance = m_Filter.output();

    return m_Classifier.distributionForInstance(newInstance);
  }
  
  /**
   * returns some information about the parameters
   * 
   * @return		information about the parameters
   */
  @Override
  protected String toStringParameters() {
    return "Filter................: " + getSpecification(getFilter()) + "\n";
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
    CollectiveEvaluationUtils.runClassifier(new FilteredCollectiveClassifier(), args);
  }
}
