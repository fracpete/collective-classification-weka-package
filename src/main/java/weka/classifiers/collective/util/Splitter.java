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
 * Splitter.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.util;

import java.io.BufferedReader;
import java.io.FileReader;

import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

/**
 * A helper class for splitting a single training set into train and test
 * set.
 * 
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $
 */
public class Splitter {
  /** the instances to split into train and test */
  protected Instances m_Instances;

  /** whether the splitting already happened */
  protected boolean m_Processed;
  
  /** The number of folds to split the training set into train and test set.
   *  E.g. 5 folds result in 20% train and 80% test set. */
  protected int m_SplitFolds;
  
  /** Whether to invert the folds, i.e., instead of taking the first fold as
   *  training set it is taken as test set and the rest as training. */
  protected boolean m_InvertSplitFolds = false;

  /** The test instances */
  protected Instances m_Testset = null;
  
  /** The training instances */
  protected Instances m_Trainset = null;
  
  /** whether to output some information during improving the classifier */
  protected boolean m_Verbose = false;

  /**
   * initializes the splitting
   * @param inst        the instances to split into train and test
   */
  public Splitter(Instances inst) {
    this(inst, 5);
  }

  /**
   * initializes the splitting
   * @param inst        the instances to split into train and test
   * @param folds       the number of folds to use for splitting
   */
  public Splitter(Instances inst, int folds) {
    this(inst, folds, false);
  }

  /**
   * initializes the splitting
   * @param inst        the instances to split into train and test
   * @param folds       the number of folds to use for splitting
   * @param invert      whether to invert the folds
   */
  public Splitter(Instances inst, int folds, boolean invert) {
    m_Instances = inst;
    // set class index
    if (m_Instances.classIndex() == -1)
      m_Instances.setClassIndex(m_Instances.numAttributes() - 1);
    
    setSplitFolds(folds);
    setInvertSplitFolds(invert);
  }
  
  /**
   * Set the percentage for splitting the train set into train and test set.
   * Use 0 for no splitting, which results in test = train.
   *
   * @param splitFolds the split percentage (1/splitFolds*100)
   */
  public void setSplitFolds(int splitFolds) {
    if (splitFolds >= 2)
      m_SplitFolds = splitFolds;
    else
      m_SplitFolds = 0;
    
    m_Processed = false;
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
   * Sets whether to use the first fold as training set (= FALSE) or as
   * test set (= TRUE).
   *
   * @param invertSplitFolds whether to invert the folding scheme
   */
  public void setInvertSplitFolds(boolean invertSplitFolds) {
    m_InvertSplitFolds = invertSplitFolds;
    m_Processed = false;
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
   * Set the verbose state.
   *
   * @param verbose the verbose state
   */
  public void setVerbose(boolean verbose) {
    m_Verbose = verbose;
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
   * returns the new Instances after applying the filter
   * 
   * @param inst the instances to apply the filter to
   * @param inverted whether to invert the folding scheme
   * @return the instances after applying the filter
   * @throws Exception if something goes wrong with the filter
   */
  protected Instances buildInstances(Instances inst, boolean inverted) 
    throws Exception {

    Instances             instTmp;
    Filter                filter;
    String[]              options;
    
    // options for filter
    if (inverted)
      options = new String[5];
    else
      options = new String[4];
    
    options[0] = "-N";
    options[1] = Integer.toString(getSplitFolds());
    options[2] = "-F";
    options[3] = "1";
    if (inverted)
      options[4] = "-V";
 
    // apply filter
    filter  = new StratifiedRemoveFolds();
    instTmp = new Instances(inst);
    filter.setInputFormat(instTmp);
    ((OptionHandler) filter).setOptions(options);
    instTmp = Filter.useFilter(instTmp, filter);
    
    return instTmp;
  }
  
  /**
   * splits the train set into train and test set if no test set was provided,
   * according to the set SplitFolds (using Filter "StratifiedRemoveFolds").
   * The new train and test can be accessed via getTrainSet() and getTestSet().
   *
   * @see StratifiedRemoveFolds
   * @see #getSplitFolds()
   * @see #getInvertSplitFolds()
   * @see #getTrainset()
   * @see #getTestset()
   * @throws Exception if anything goes wrong with the Filter
   */
  protected void process() throws Exception {
    String            tmpPercentage;
        
    // already processed? -> exit
    if (m_Processed)
      return;
    
    if (m_SplitFolds != 0) {
      if (!getInvertSplitFolds())
	tmpPercentage = Utils.roundDouble((double) 100 / getSplitFolds(), 1) 
	                + "/" 
	                + Utils.roundDouble((double) 100 / getSplitFolds() * (getSplitFolds() - 1), 1);
      else
	tmpPercentage = Utils.roundDouble((double) 100 / getSplitFolds() * (getSplitFolds() - 1), 1) 
	                + "/" 
	                + Utils.roundDouble((double) 100 / getSplitFolds(), 1);
	    
      System.out.println(
	  "WARNING: No test file provided! \n" 
	  + "         -> splitting training file with " 
	  + getSplitFolds() + " folds (" + tmpPercentage + ")!");

      m_Trainset = new Instances(
	  buildInstances(m_Instances,  getInvertSplitFolds()));
      m_Testset  = new Instances(
	  buildInstances(m_Instances, !getInvertSplitFolds()));
    }
    else {
      m_Trainset = new Instances(m_Instances);
      m_Testset  = new Instances(m_Instances);
      
      System.out.println(
	  "WARNING: No splitting will be performed, test = train!");
    }
    
    if (getVerbose())
      System.out.println("Numbers: " + m_Instances.numInstances() + " -> " 
        + m_Trainset.numInstances() + "/" + m_Testset.numInstances());
    
    m_Processed = true;
  }

  /**
   * returns the training set
   * 
   * @return 		the training set
   * @throws Exception	if something goes wrong
   */
  public Instances getTrainset() throws Exception {
    process();
    
    return m_Trainset;
  }

  /**
   * returns the test set
   * 
   * @return		the test set
   * @throws Exception	if something goes wrong
   */
  public Instances getTestset() throws Exception {
    process();
    
    return m_Testset;
  }

  /**
   * For testing only. Takes an ARFF filename as first parameter, as second
   * the number of folds, as third whether to invert and as fourth whether
   * to verbose.
   * 
   * @param args	the commandline arguments
   * @throws Exception	if something goes wrong, e.g., file not found
   */
  public static void main(String[] args) throws Exception {
    Instances     inst;
    Splitter      splitter;
    int           folds;
    boolean       invert;
    boolean       verbose;

    // nothing provided
    if (args.length == 0) {
      System.out.println("Usage: " + Splitter.class.getName() 
          + " <arff-file> [folds(int)] [inverted(true|false)] [verbose(true|false)]");
      return;
    }
    
    // load file
    inst = new Instances(new BufferedReader(new FileReader(args[0])));

    // get folds
    if (args.length > 1)
      folds = Integer.parseInt(args[1]);
    else
      folds = 5;
    
    // get inversion
    if (args.length > 2)
      invert = Boolean.getBoolean(args[2]);
    else
      invert = false;
    
    // get verbose
    if (args.length > 3)
      verbose = Boolean.getBoolean(args[3]);
    else
      verbose = false;
    
    // split file
    splitter = new Splitter(inst);
    splitter.setSplitFolds(folds);
    splitter.setInvertSplitFolds(invert);
    splitter.setVerbose(verbose);
    System.out.println("\nTraining:\n\n" + splitter.getTrainset());
    System.out.println("\nTest:\n\n"     + splitter.getTestset());
  }
}
