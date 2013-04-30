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

/**
 * CollectiveEvaluation.java
 * Copyright (C) 2013 University of Waikato, Hamilton, New Zealand
 */
package weka.classifiers;

import weka.classifiers.collective.CollectiveClassifier;
import weka.core.Instances;

/**
 * Wrapper class around {@link weka.classifiers.collective.evaluation.Evaluation}.
 * 
 * @author  fracpete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2024 $
 */
public class CollectiveEvaluation
  extends Evaluation {

  /** for serialization. */
  private static final long serialVersionUID = 430772074540191082L;

  /**
   * Initializes all the counters for the evaluation. Use
   * <code>useNoPriors()</code> if the dataset is the test set and you can't
   * initialize with the priors from the training set via
   * <code>setPriors(Instances)</code>.
   * 
   * @param data set of training instances, to get some header information and
   *          prior class distribution information
   * @throws Exception if the class is not defined
   */
  public CollectiveEvaluation(Instances data) throws Exception {
    this(data, null);
  }

  /**
   * Initializes all the counters for the evaluation and also takes a cost
   * matrix as parameter. Use <code>useNoPriors()</code> if the dataset is the
   * test set and you can't initialize with the priors from the training set via
   * <code>setPriors(Instances)</code>.
   * 
   * @param data set of training instances, to get some header information and
   *          prior class distribution information
   * @param costMatrix the cost matrix---if null, default costs will be used
   * @throws Exception if cost matrix is not compatible with data, the class is
   *           not defined or the class is numeric
   */
  public CollectiveEvaluation(Instances data, CostMatrix costMatrix) throws Exception {
    super(data, costMatrix);
    m_delegate = new weka.classifiers.collective.evaluation.Evaluation(data, costMatrix);
  }

  /**
   * Evaluates a classifier with the options given in an array of strings.
   * 
   * @param classifierString class of machine learning classifier as a string
   * @param options the array of string containing the options
   * @throws Exception if model could not be evaluated successfully
   * @return a string describing the results
   */
  public static String evaluateModel(String classifierString, String[] options) throws Exception {
    return weka.classifiers.collective.evaluation.Evaluation.evaluateModel(classifierString, options);
  }
  
  /**
   * Evaluates a classifier with the options given in an array of strings.
   * 
   * @param classifier machine learning classifier
   * @param options the array of string containing the options
   * @throws Exception if model could not be evaluated successfully
   * @return a string describing the results
   */
  public static String evaluateModel(CollectiveClassifier classifier, String[] options) throws Exception {
    return weka.classifiers.collective.evaluation.Evaluation.evaluateModel(classifier, options);
  }
}
