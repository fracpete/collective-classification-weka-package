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
 * CollectiveTreeModel.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.trees.model;

/**
 * A generic class for storing a collective tree model. It extends the default
 * decision tree model.
 *
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $
 */
public class CollectiveTreeModel 
  extends DecisionTreeModel {

  /** for serialization */
  private static final long serialVersionUID = 2234286451642253295L;

  /**
   * Creates a tree in which any node can have children.
   * 
   * @param root	the root node
   */
  public CollectiveTreeModel(CollectiveTreeNode root) {
    super(root);
    setTreeName("CollectiveTree");
  }
}
