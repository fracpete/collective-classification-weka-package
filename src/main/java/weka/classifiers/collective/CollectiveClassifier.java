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
 * CollectiveClassifier.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.OptionHandler;

/** 
 * Interface that all collective classifiers implement.
 *
 * @author FracPete (fracpete at cs dot waikato dot ac dot nz)
 * @version $Revision: 2019 $
 */
public interface CollectiveClassifier 
  extends Classifier, OptionHandler {
  
  /**
   * sets the instances used for testing
   * 
   * @param value the instances used for testing
   */
  public void setTestSet(Instances value);
  
  /**
   * returns the Test Set
   *
   * @return the Test Set
   */
  public Instances getTestSet();
  
  /**
   * returns the Training Set
   *
   * @return the Training Set
   */
  public Instances getTrainingSet();
  
  /**
   * resets the classifier
   */
  public void reset();
  
  /**
   * Set the verbose state.
   *
   * @param value the verbose state
   */
  public void setVerbose(boolean value);
  
  /**
   * Gets the verbose state
   *
   * @return the verbose state
   */
  public boolean getVerbose();
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String verboseTipText();
  
  /**
   * Set the percentage for splitting the train set into train and test set. 
   * Use 0 for no splitting, which results in test = train.
   *
   * @param value the split percentage (1/splitFolds*100)
   */
  public void setSplitFolds(int value);
  
  /**
   * Gets the split percentage for splitting train set into train and test set
   *
   * @return the split percentage (1/splitFolds*100)
   */
  public int getSplitFolds();
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String splitFoldsTipText();
  
  /**
   * Sets whether to use the first fold as training set (= FALSE) or as
   * test set (= TRUE).
   *
   * @param value whether to invert the folding scheme
   */
  public void setInvertSplitFolds(boolean value);
  
  /**
   * Gets whether to use the first fold as training set (= FALSE) or as
   * test set (= TRUE)
   *
   * @return whether to invert the folding scheme
   */
  public boolean getInvertSplitFolds();
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String invertSplitFoldsTipText();

  /**
   * Whether to use the labels of the test set to output more statistics
   * (not used for learning and only for debugging purposes)
   * 
   * @param value	if true, more statistics are output
   */
  public void setUseInsight(boolean value);

  /**
   * Returns whether we use the labels of the test set to output some more
   * statistics.
   * 
   * @return		true if more statistics are output
   */
  public boolean getUseInsight();
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String useInsightTipText();
  
  /**
   * Method for building this classifier.
   * 
   * @param training	the training instances
   * @param test	the test instances
   * @throws Exception	if somethong goes wrong
   */
  public void buildClassifier(Instances training, Instances test) 
    throws Exception;
}
