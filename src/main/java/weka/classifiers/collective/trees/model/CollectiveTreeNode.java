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
 * CollectiveTreeNode.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.trees.model;

import weka.core.Utils;

/**
 * A generic class for storing a node in a CollectiveTree. It extends the
 * default DecisionTreeNode with some further fields and methods.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $
 */
public class CollectiveTreeNode
  extends DecisionTreeNode {

  /** for serialization */
  private static final long serialVersionUID = -3254971946913047468L;

  /** The class distribution of the test set. */
  protected double[][] m_TestDistribution = null;

  /** The proportions of the test instances going down each branch. */
  protected double[] m_TestProp = null;
    
  /**
   * initializes the node, adds itself to the parents children if parent
   * is not null.
   * 
   * @param parent      the parent of this node, null if its the root node
   */
  public CollectiveTreeNode(DecisionTreeNode parent) {
    super(parent);
  }

  /**
   * returns the class distribution from the data.
   * 
   * @return		the distribution
   */
  public double[][] getTestDistribution() {
    return m_TestDistribution;
  }

  /**
   * sets the class distribution from the data.
   * 
   * @param dist	the distribution
   */
  public void setTestDistribution(double[][] dist) {
    if (dist == null) {
      m_TestDistribution = null;
    }
    else {
      m_TestDistribution = new double[dist.length][dist[0].length];
      
      for (int i = 0; i < dist.length; i++)
        for (int j = 0; j < dist[0].length; j++)
          m_TestDistribution[i][j] = dist[i][j];
    }
  }

  /**
   * returns the proportions of instances going down each branch.
   * 
   * @return		the test proportions
   */
  public double[] getTestProportions() {
    return m_TestProp;
  }

  /**
   * sets the proportions of instances going down each branch.
   * 
   * @param prop	the proportions
   */
  public void setTestProportions(double[] prop) {
    if (prop == null) {
      m_TestProp = null;
    }
    else {
      m_TestProp = new double[prop.length];
      
      for (int i = 0; i < prop.length; i++)
        m_TestProp[i] = prop[i];
    }
  }

  /**
   * returns detailed information about the node
   * 
   * @return		a string representation of the node
   */
  @Override
  public String toStringNode() {
    String        result;

    result  = super.toStringNode();
    result += "Dist.-Test..: " + Utils.arrayToString(getTestDistribution()) + "\n";
    result += "Props.-Test.: " + Utils.arrayToString(getTestProportions()) + "\n";
    
    return result;
  }
}


