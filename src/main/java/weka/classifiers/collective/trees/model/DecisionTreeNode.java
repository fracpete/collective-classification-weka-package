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
 * DecisionTreeNode.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.trees.model;

import java.io.Serializable;

import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreeNode;

import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * A generic class for storing a node in a decision tree. It extends the
 * default mutable tree node with some WEKA specific methods and fields.
 *
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $
 */
public class DecisionTreeNode
  extends DefaultMutableTreeNode
  implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = -5616540811927424614L;

  /** Whether to print any debug information (&gt; 0) */
  protected int m_DebugLevel = 0;

  /** The information about the dataset, but no data. */
  protected Instances m_Info = null;

  /** The attribute to split on. */
  protected int m_Attribute = -1;
    
  /** The split point. */
  protected double m_SplitPoint = Double.NaN;

  /** The distribution of the nominal values into the different branches. */
  protected double[][] m_NominalSplit = null;
    
  /** The class distribution from the training data. */
  protected double[][] m_Distribution = null;

  /** The proportions of training instances going down each branch. */
  protected double[] m_Prop = null;

  /** Class probabilities from the training data. */
  protected double[] m_ClassProbs = null;
    
  /**
   * initializes the node, adds itself to the parents children if parent
   * is not null.
   * 
   * @param parent      the parent of this node, null if its the root node
   */
  public DecisionTreeNode(DecisionTreeNode parent) {
    super();

    if (parent != null)
      parent.add(this);
  }

  /**
   * Returns the current debug level (0 means off).
   * 
   * @return		the debug level
   */
  public int getDebugLevel() {
    if (isRoot())
      return m_DebugLevel;
    else
      return ((DecisionTreeNode) getRoot()).getDebugLevel();
  }

  /**
   * Sets the new debug level (0 means off, the higher, the more output).
   * 
   * @param level	the debug level
   */
  public void setDebugLevel(int level) {
    if (level >= 0) {
      if (isRoot())
        m_DebugLevel = level;
      else
        ((DecisionTreeNode) getRoot()).setDebugLevel(level);
    }
  }

  /**
   * returns the information about the dataset
   * 
   * @return		the information (just the header of the data)
   */
  public Instances getInformation() {
    if (isRoot())
      return m_Info;
    else
      return ((DecisionTreeNode) getRoot()).getInformation();
  }

  /**
   * sets the information about the dataset, i.e., it creates a copy of the
   * header information.
   * 
   * @param info	the header of the data
   */
  public void setInformation(Instances info) {
    if (isRoot())
      m_Info = new Instances(info, 0);
    else
      m_Info = ((DecisionTreeNode) getRoot()).getInformation();
  }

  /**
   * returns the attribute the split was performed on.
   * 
   * @return		the attribute index
   */
  public int getAttribute() {
    return m_Attribute;
  }

  /**
   * returns the reference of the attribute the split was performed on. NULL if
   * no attribute was set.
   * 
   * @return		the attribute
   */
  public Attribute getAttributeRef() {
    if (getAttribute() > -1)
      return getInformation().attribute(getAttribute());
    else
      return null;
  }

  /**
   * sets the attribute the split was performed on.
   * 
   * @param att		the attribute
   */
  public void setAttribute(int att) {
    m_Attribute = att;
  }

  /**
   * returns the split point.
   * 
   * @return		the split point
   */
  public double getSplitPoint() {
    return m_SplitPoint;
  }

  /**
   * sets the split point.
   * 
   * @param split	the split point
   */
  public void setSplitPoint(double split) {
    m_SplitPoint = split;
  }

  /**
   * returns the nominal values that go into each branch.
   * 
   * @return		the nominal values
   */
  public double[][] getNominalSplit() {
    return m_NominalSplit;
  }

  /**
   * sets the nominal values that go into each branch.
   * 
   * @param split	the nominal values
   */
  public void setNominalSplit(double[][] split) {
    if (split == null) {
      m_NominalSplit = null;
    }
    else {
      m_NominalSplit = new double[split.length][0];
      
      for (int i = 0; i < split.length; i++) {
        m_NominalSplit[i] = new double[split[i].length];
        for (int j = 0; j < split[i].length; j++)
          m_NominalSplit[i][j] = split[i][j];
      }
    }
  }

  /**
   * returns the class distribution from the data.
   * 
   * @return		the distribution
   */
  public double[][] getDistribution() {
    return m_Distribution;
  }

  /**
   * sets the class distribution from the data.
   * 
   * @param dist	the distribution
   */
  public void setDistribution(double[][] dist) {
    if (dist == null) {
      m_Distribution = null;
    }
    else {
      m_Distribution = new double[dist.length][0];
      
      for (int i = 0; i < dist.length; i++) {
        m_Distribution[i] = new double[dist[i].length];
        for (int j = 0; j < dist[i].length; j++)
          m_Distribution[i][j] = dist[i][j];
      }
    }
  }

  /**
   * returns the proportions of instances going down each branch.
   * 
   * @return		the proportions
   */
  public double[] getProportions() {
    return m_Prop;
  }

  /**
   * sets the proportions of instances going down each branch.
   * 
   * @param prop	the proportions
   */
  public void setProportions(double[] prop) {
    if (prop == null) {
      m_Prop = null;
    }
    else {
      m_Prop = new double[prop.length];
      
      for (int i = 0; i < prop.length; i++)
        m_Prop[i] = prop[i];
    }
  }

  /**
   * returns the class probabilities.
   * 
   * @return		the class probabilities
   */
  public double[] getClassProbabilities() {
    return m_ClassProbs;
  }

  /**
   * sets the class probabilities.
   * 
   * @param probs	the class probabilities
   */
  public void setClassProbabilities(double[] probs) {
    if (probs == null) {
      m_ClassProbs = null;
    }
    else {
      m_ClassProbs = new double[probs.length];

      for (int i = 0; i < probs.length; i++)
        m_ClassProbs[i] = probs[i];
    }
  }

  /**
   * sets the class probabilities based on the given data
   * 
   * @param data	the data to get the class probabilities from
   */
  public void setClassProbabilities(Instances data) {
    AttributeStats	stats;
    int			total;
    int			i;
    
    stats = data.attributeStats(data.classIndex());
    total = Utils.sum(stats.nominalCounts);
    m_ClassProbs = new double[data.classAttribute().numValues()];
    for (i = 0; i < m_ClassProbs.length; i++)
      m_ClassProbs[i] = (double) stats.nominalCounts[i] / (double) total;
  }

  /**
   * same as getChildAt(int), but automatically casts to DecisionTreeNode
   * @param index       the child to retrieve
   * @return            the child at the given position
   * @throws ArrayIndexOutOfBoundsException if index is out of bounds
   */
  public DecisionTreeNode getNodeAt(int index) 
    throws ArrayIndexOutOfBoundsException {
    return (DecisionTreeNode) getChildAt(index);
  }

  /**
   * Returns the size of the subtree, including the node itself. In case of
   * a leaf i.e., 1
   * 
   * @return		the size of the subtree, including itself
   */
  public int size() {
    int       size;
    int       i;

    size = 1;

    for (i = 0; i < getChildCount(); i++)
      size += getNodeAt(i).size();

    return size;
  }
  
  /**
   * Computes class distribution of an instance using the decision tree.
   * 
   * @param instance	the instance to compute the distribution for
   * @return		the class distribution
   * @throws Exception	if something goes wrong
   */
  public double[] distributionForInstance(Instance instance) throws Exception {
    
    double[] returnedDist = null;
    
    if (getAttribute() > -1) {
      // Node is not a leaf
      if (instance.isMissing(getAttribute())) {
        if (getDebugLevel() > 0)
          System.out.println(toStringNode());

	// Value is missing
	returnedDist = new double[getInformation().numClasses()];
        
	// Split instance up
	for (int i = 0; i < getChildCount(); i++) {
	  double[] help = getNodeAt(i).distributionForInstance(instance);
          if (getDebugLevel() > 1)
            System.out.println("help: " + Utils.arrayToString(help));
	  if (help != null) {
	    for (int j = 0; j < help.length; j++) {
	      returnedDist[j] += m_Prop[i] * help[j];
	    }
	  }
	}
        if (getDebugLevel() > 1)
          System.out.println(   "--> returnedDist: " 
                              + Utils.arrayToString(returnedDist));
      } 
      else if (getInformation().attribute(getAttribute()).isNominal()) {
	// For nominal attributes
        int branch = 0;

        // branch for each nominal value?
        if (getNominalSplit() == null) {
          branch = (int) instance.value(getAttribute());
        }
        else {
          // determine the branch we have to go down
          for (int i = 0; i < getNominalSplit().length; i++) {
            for (int n = 0; n < getNominalSplit()[i].length; n++) {
              if (Utils.eq(instance.value(getAttribute()), 
                           getNominalSplit()[i][n])) {
                branch = i;
                break;
              }
            }
          }
        }

        returnedDist = getNodeAt(branch).distributionForInstance(instance);
      } 
      else {
	// For numeric attributes
	if (Utils.sm(instance.value(getAttribute()), getSplitPoint())) {
	  returnedDist = getNodeAt(0).distributionForInstance(instance);
	} 
        else {
	  returnedDist = getNodeAt(1).distributionForInstance(instance);
	}
      }
    }

    if ((getAttribute() == -1) || (returnedDist == null)) {
      // Node is a leaf or successor is empty
      return getClassProbabilities();
    } 
    else {
      return returnedDist;
    }
  }

  /**
   * returns the attribute name it was split on
   * 
   * @return	the attribute name
   */
  @Override
  public String toString() {
    if (getAttribute() > -1)
      return getAttributeRef().name();
    else
      return "???";
  }

  /**
   * returns detailed information about the node
   * 
   * @return	the node in a string representation
   */
  public String toStringNode() {
    String          result;
    TreeNode[]      path;
    int             i;

    result  = "--> Node\n";
    result += "Path........: ";
    path    = getPath();
    for (i = 0; i < path.length; i++) {
      if (i > 0)
        result += " -> ";
      result += path[i].toString();
    }
    result += "\n";
    
    // path
    
    // fields
    if (getAttributeRef() != null)
      result += "Attribute...: " + getAttributeRef().name() 
                                 + " (" + getAttribute() 
                                 + ", nominal=" + getAttributeRef().isNominal()
                                 + ")\n";
    else
      result += "Attribute...: -not set-\n";
    result += "SplitPoint..: " + getSplitPoint() + "\n";
    result += "NominalSplit: " + Utils.arrayToString(getNominalSplit()) + "\n";
    result += "Distribution: " + Utils.arrayToString(getDistribution()) + "\n";
    result += "Proportions.: " + Utils.arrayToString(getProportions()) + "\n";
    result += "ClassProbs..: " + Utils.arrayToString(getClassProbabilities()) 
                               + "\n";

    return result;
  }
}


