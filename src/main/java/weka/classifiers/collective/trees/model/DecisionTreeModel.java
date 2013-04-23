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
 * DecisionTreeModel.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.trees.model;

import java.io.Serializable;

import javax.swing.tree.DefaultTreeModel;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * A generic class for storing a decision tree model. It extends the default
 * tree model with some WEKA specific methods and fields.
 *
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $
 */
public class DecisionTreeModel 
  extends DefaultTreeModel 
  implements Serializable {

  /** for serializatin */
  private static final long serialVersionUID = -6214204368479074117L;

  /** the name of the tree */
  protected String m_TreeName = "Decision-Tree";

  /** 
   * For getting a unique ID when outputting the tree source
   * (hashcode isn't guaranteed unique) 
   */
  protected static long PRINTED_NODES = 0;

  /**
   * Creates a tree in which any node can have children.
   * 
   * @param root	the root node
   */
  public DecisionTreeModel(DecisionTreeNode root) {
    super(root);
  }

  /**
   * Returns the current debug level (0 means off).
   * 
   * @return		the debug level
   */
  public int getDebugLevel() {
    return getRootNode().getDebugLevel();
  }

  /**
   * Sets the new debug level (0 means off, the higher, the more output).
   * 
   * @param level	the debug level to use
   */
  public void setDebugLevel(int level) {
    getRootNode().setDebugLevel(level);
  }

  /**
   * sets the name of the tree.
   * 
   * @param name	the name of the tree
   */
  public void setTreeName(String name) {
    m_TreeName = name;
  }

  /**
   * returns the currently set tree name.
   * 
   * @return		the name of the tree
   */
  public String getTreeName() {
    return m_TreeName;
  }

  /**
   * returns the root node, like getRoot(), but casts it to DecisionTreeNode
   * 
   * @return		the root node
   */
  public DecisionTreeNode getRootNode() {
    return (DecisionTreeNode) getRoot();
  }

  /**
   * returns the number of nodes in the tree
   * 
   * @return		the size of the tree, i.e., the number of nodes
   */
  public int size() {
    return getRootNode().size();
  }

  /**
   * returns the distribution for the given instance
   *
   * @param instance	the instance to determine the distribution for
   * @return		the distribution for the given instance
   * @throws Exception	if something goes wrong
   */
  public double[] distributionForInstance(Instance instance) throws Exception {
    return ((DecisionTreeNode) getRoot()).distributionForInstance(instance);
  }

  /**
   * returns the information about the dataset, not the data itself
   * 
   * @return		the information (just the header of the data)
   */
  public Instances getInformation() {
    if (getRoot() == null)
      return null;
    else
      return ((DecisionTreeNode) getRoot()).getInformation();
  }
  
  /**
   * Outputs one node for graph.
   * 
   * @param text	the buffer to add the data to
   * @param num		the node number
   * @param node	the node
   * @return		the new node number
   * @throws Exception	if something goes wrong
   */
  public int toGraph(StringBuffer text, int num, DecisionTreeNode node) 
    throws Exception {

    double[] classprobs = node.getClassProbabilities();
    int att = node.getAttribute();
    double splitpoint = node.getSplitPoint();
    Instances info = node.getInformation();
    
    int maxIndex = Utils.maxIndex(classprobs);
    String classValue = info.classAttribute().value(maxIndex);
    
    num++;
    if (att == -1) {
      text.append("N" + Integer.toHexString(hashCode()) +
		  " [label=\"" + num + ": " + classValue + "\"" +
		  "shape=box]\n");
    }
    else {
      text.append("N" + Integer.toHexString(hashCode()) +
		  " [label=\"" + num + ": " + classValue + "\"]\n");
      for (int i = 0; i < node.getChildCount(); i++) {
	text.append("N" + Integer.toHexString(hashCode()) 
		    + "->" + 
		    "N" + Integer.toHexString(node.getNodeAt(i).hashCode())  +
		    " [label=\"" + info.attribute(att).name());
	if (info.attribute(att).isNumeric()) {
	  if (i == 0) {
	    text.append(" < " +
			Utils.doubleToString(splitpoint, 2));
	  } 
          else {
	    text.append(" >= " +
			Utils.doubleToString(splitpoint, 2));
	  }
	} 
        else {
          // split for every nominal value?
          if (node.getNominalSplit() == null) {
            text.append( " = " +
                        info.attribute(att).value(i));
          }
          else {
            text.append(" = [");
            for (int n = 0; n < node.getNominalSplit()[i].length; n++) {
              if (n > 0)
                text.append(", ");
              text.append(
                  info.attribute(att).value((int) node.getNominalSplit()[i][n]));
            }
            text.append("]");
          }
	}
	text.append("\"]\n");
	num = toGraph(text, num, node.getNodeAt(i));
      }
    }
    
    return num;
  }

  /**
   * the decision tree as a graph
   * 
   * @return		the tree as graph
   */
  public String getGraph() {
    String        result;
    StringBuffer  resultBuff;

    try {
      resultBuff = new StringBuffer();
      toGraph(resultBuff, 0, getRootNode());
      result =   "digraph Tree {\n" 
               + "edge [style=bold]\n" 
               + resultBuff.toString()
	       + "\n}\n";
    } 
    catch (Exception e) {
      result = null;
    }

    return result;
  }

  /**
   * Gets the next unique node ID.
   *
   * @return the next unique node ID.
   */
  protected static long nextID() {
    return PRINTED_NODES++;
  }

  /**
   * restes the ID counter
   */
  protected static void resetID() {
    PRINTED_NODES = 0;
  }

 /**
  * Returns a string containing java source code equivalent to the test
  * made at this node. The instance being tested is called "i". This
  * routine assumes to be called in the order of branching, enabling us to
  * set the >= condition test (the last one) of a numeric splitpoint 
  * to just "true" (because being there in the flow implies that the 
  * previous less-than test failed).
  *
  * @param node the current node
  * @param index index of the value tested
  * @return a value of type 'String'
  */
  protected final String sourceExpression(DecisionTreeNode node, int index) {
    int m_Attribute = node.getAttribute();
    Instances m_Info = node.getInformation();
    double m_SplitPoint = node.getSplitPoint();
    
    StringBuffer expr = null;
    if (index < 0) {
      return "i[" + m_Attribute + "] == null";
    }
    if (m_Info.attribute(m_Attribute).isNominal()) {
      expr = new StringBuffer("i[");
      expr.append(m_Attribute).append("]");
      expr.append(".equals(\"").append(m_Info.attribute(m_Attribute)
              .value(index)).append("\")");
    } else {
      expr = new StringBuffer("");
      if (index == 0) {
        expr.append("((Double)i[")
          .append(m_Attribute).append("]).doubleValue() < ")
          .append(m_SplitPoint);
      } else {
        expr.append("true");
      }
    }
    return expr.toString();
  }

 /**
  * Returns source code for the tree as if-then statements. The 
  * class is assigned to variable "p", and assumes the tested 
  * instance is named "i". The results are returned as two stringbuffers: 
  * a section of code for assignment of the class, and a section of
  * code containing support code (eg: other support methods).
  *
  * TODO: If the outputted source code encounters a missing value
  * for the evaluated attribute, it stops branching and uses the 
  * class distribution of the current node to decide the return value. 
  * This is unlike the behaviour of distributionForInstance(). 
  *
  * TODO: nominal values that don't branch for each value, but for sets of
  * values???
  *
  * @param className the classname that this static classifier has
  * @param node the current node 
  * @return an array containing two stringbuffers, the first string containing
  * assignment code, and the second containing source for support code.
  * @exception Exception if something goes wrong
  */
  protected StringBuffer[] toSource(String className, DecisionTreeNode node) 
    throws Exception {
  
    DecisionTreeNode parent = (DecisionTreeNode) node.getParent();
    StringBuffer [] result = new StringBuffer[2];
    double[] currentProbs;
    int m_Attribute = node.getAttribute();
    Instances m_Info = node.getInformation();

    if(node.getClassProbabilities() == null)
      currentProbs = parent.getClassProbabilities();
    else
      currentProbs = node.getClassProbabilities();

    long printID = nextID();

    // Is this a leaf?
    if (m_Attribute == -1) {
      result[0] = new StringBuffer("	p = ");
      if(m_Info.classAttribute().isNumeric()) {
        result[0].append(currentProbs[0]);
      }
      else {
        result[0].append(Utils.maxIndex(currentProbs));
      }
      result[0].append(";\n");
      result[1] = new StringBuffer("");
    } 
    else {
      StringBuffer text = new StringBuffer("");
      StringBuffer atEnd = new StringBuffer("");

      text.append("  static double N")
        .append(Integer.toHexString(this.hashCode()) + printID)
        .append("(Object[] i) {\n")
        .append("    double p = Double.NaN;\n");

      text.append("    /* " + m_Info.attribute(m_Attribute).name() + " */\n");
      // Missing attribute?
      text.append("    if (" + this.sourceExpression(node, -1) + ") {\n")
        .append("      p = ");
      if (m_Info.classAttribute().isNumeric())
        text.append(currentProbs[0] + ";\n");
      else
        text.append(Utils.maxIndex(currentProbs) + ";\n");
      text.append("    } ");
      
      // Branching of the tree
      for (int i = 0; i < node.getChildCount(); i++) {
        text.append(" else if (" + this.sourceExpression(node, i) + ") {\n");
        // Is the successor a leaf?
        if (node.getNodeAt(i).getAttribute() == -1) {
          double[] successorProbs = node.getNodeAt(i).getClassProbabilities();
          if (successorProbs == null)
            successorProbs = node.getClassProbabilities();
          text.append("      p = ");
          if (m_Info.classAttribute().isNumeric()) {
            text.append(successorProbs[0] + ";\n");
          } 
          else {
            text.append(Utils.maxIndex(successorProbs) + ";\n");
          }
        } 
        else {
          StringBuffer [] sub = toSource(className, node.getNodeAt(i));
          text.append("" + sub[0]);
          atEnd.append("" + sub[1]);
        }
        text.append("    } ");
        if (i == node.getChildCount() - 1) {
          text.append("\n");
        }
      }

      text.append("    return p;\n  }\n");

      result[0] = new StringBuffer("    p = " + className + ".N");
      result[0].append(Integer.toHexString(this.hashCode()) + printID)
        .append("(i);\n");
      result[1] = text.append("" + atEnd);
    }
    
    return result;
  }

  /**
   * Returns the tree as if-then statements.
   *
   * @param className	the classname for the generated code
   * @return 		the tree as a Java if-then type statement
   */
  public String getSource(String className) {
    String result;
    StringBuffer[] source;

    try {
      source = toSource(className, getRootNode());
      result = 
          "class " + className + " {\n\n"
        + "  public static double classify(Object[] i)\n"
        + "    throws Exception {\n\n"
        + "    double p = Double.NaN;\n"
        + source[0]  // Assignment code
        + "    return p;\n"
        + "  }\n"
        + source[1]  // Support code
        + "}\n";
    }
    catch (Exception e) {
      e.printStackTrace();
      result = "";
    }

    return result;
  }

  /**
   * Outputs a leaf.
   * 
   * @param node	the node to output
   * @return		the leaf as string
   * @throws Exception	if something goes wrong
   */
  protected String leafString(DecisionTreeNode node) throws Exception {
    double[][] dist = node.getDistribution();
    Instances info = node.getInformation();
    int maxIndex = Utils.maxIndex(dist[0]);

    return " : " + info.classAttribute().value(maxIndex) + 
      " (" + Utils.doubleToString(Utils.sum(dist[0]), 2) + "/" + 
      Utils.doubleToString((Utils.sum(dist[0]) - 
			    dist[0][maxIndex]), 2) + ")";
  }
  
  /**
   * Recursively outputs the tree.
   * 
   * @param node	the node to output
   * @return		a string representation of the node
   */
  protected String toString(DecisionTreeNode node) {
    int level = node.getLevel();
    int att = node.getAttribute();
    Instances info = node.getInformation();
    double splitpoint = node.getSplitPoint();
    
    try {
      StringBuffer text = new StringBuffer();
      
      if (att == -1) {
	
	// Output leaf info
	return leafString(node);
      } else if (info.attribute(att).isNominal()) {
	
	// For nominal attributes
	for (int i = 0; i < node.getChildCount(); i++) {
	  text.append("\n");
	  for (int j = 0; j < level; j++) {
	    text.append("|   ");
	  }
          // split for every nominal value?
          if (node.getNominalSplit() == null) {
            text.append(info.attribute(att).name() + " = " +
                        info.attribute(att).value(i));
          }
          else {
            text.append(info.attribute(att).name() + " = [");
            for (int n = 0; n < node.getNominalSplit()[i].length; n++) {
              if (n > 0)
                text.append(", ");
              text.append(
                  info.attribute(att).value((int) node.getNominalSplit()[i][n]));
            }
            text.append("]");
          }
	  text.append(toString(node.getNodeAt(i)));
	}
      } else {
	
	// For numeric attributes
	text.append("\n");
	for (int j = 0; j < level; j++) {
	  text.append("|   ");
	}
	text.append(info.attribute(att).name() + " < " +
		    Utils.doubleToString(splitpoint, 2));
	text.append(toString(node.getNodeAt(0)));
	text.append("\n");
	for (int j = 0; j < level; j++) {
	  text.append("|   ");
	}
	text.append(info.attribute(att).name() + " >= " +
		    Utils.doubleToString(splitpoint, 2));
	text.append(toString(node.getNodeAt(1)));
      }
      
      return text.toString();
    } catch (Exception e) {
      e.printStackTrace();
      return getTreeName() + ": tree can't be printed";
    }
  } 

  /**
   * the decision tree in string representation
   * 
   * @return		the tree as string
   */
  public String getString() {
    return   "\n" 
           + getTreeName() + "\n" 
           + getTreeName().replaceAll(".", "=") + "\n" 
           + toString(getRootNode()) + "\n\n" 
           + "Size of the tree : " + size();
  }

  /**
   * the decision tree in string representation
   * 
   * @return		the tree as string
   */
  @Override
  public String toString() {
    return getString();
  }
}
