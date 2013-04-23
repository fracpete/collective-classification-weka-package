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
 * RestartableCollectiveClassifier.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective;

/** 
 * Interface for restartable collective classifiers, i.e., classifiers that
 * use restarts and iterations to improve themselves.
 *
 * @author FracPete (fracpete at cs dot waikato dot ac dot nz)
 * @version $Revision: 2019 $
 */
public interface RestartableCollectiveClassifier 
  extends Comparable, CollectiveClassifier {
  
  /**
   * Sets the number of iterations
   * 
   * @param value	the number of iterations
   */
  public void setNumIterations(int value);
  
  /**
   * Gets the number of iterations
   *
   * @return 		the number of iterations
   */
  public int getNumIterations();
  
  /**
   * Returns the tip text for this property
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String numIterationsTipText();
  
  /**
   * Sets the number of restarts
   * 
   * @param value	the number of restarts
   */
  public void setNumRestarts(int value);
  
  /**
   * Gets the number of restarts
   *
   * @return 		the number of restarts
   */
  public int getNumRestarts();
  
  /**
   * Returns the tip text for this property
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String numRestartsTipText();
  
  /**
   * returns the run in which the last improvement happened
   * 
   * @return		the run in which the last improvement happened
   */
  public int getLastRestart();
  
  /**
   * returns the iteration in which the last improvement happened
   * 
   * @return		the iteration in which the last improvement happened
   */
  public int getLastIteration();
  
  /**
   * returns whether the classifier could be improved during the runs/iterations
   * 
   * @return		true if the classifier could be improved
   */
  public boolean classifierImproved();
  
  /**
   * initializes the labels
   * 
   * @throws Exception if initialization fails
   */
  public void initializeLabels() throws Exception;
  
  /**
   * flips the labels
   * 
   * @throws Exception if flipping fails
   */
  public void flipLabels() throws Exception;
}
