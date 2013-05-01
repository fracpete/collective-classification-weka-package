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
 * DataGenerator.java
 * Copyright (C) 2013 University of Waikato, Hamilton, New Zealand
 */
package weka.gui.explorer;

import java.util.ArrayList;

import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.gui.visualize.Plot2D;
import weka.gui.visualize.PlotData2D;

/**
 * Helper class for generating visualization data.
 *
 * @author  fracpete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 6925 $
 */
public class DataGenerator {

  /** the underlying Evaluation object. */
  protected Evaluation m_Evaluation;

  /** the underlying data. */
  protected Instances m_PlotInstances;

  /** for storing the plot shapes. */
  protected FastVector m_PlotShapes;

  /** for storing the plot sizes. */
  protected FastVector m_PlotSizes;

  /** whether the data has already been processed. */
  protected boolean m_Processed;

  /**
   * Initializes the generator.
   *
   * @param eval	the Evaluation object to use
   */
  public DataGenerator(Evaluation eval) {
    super();

    m_Evaluation = eval;
    m_Processed  = false;
  }

  /**
   * Scales the errors.
   *
   * @param data	the data containing the errors to scale
   * @return 		the scaled errors
   */
  public ArrayList scale(ArrayList data) {
    ArrayList	result;
    double 	maxErr;
    double 	minErr;
    double 	err;
    int		i;
    Double 	errd;
    double 	temp;

    result = new ArrayList();
    maxErr = Double.NEGATIVE_INFINITY;
    minErr = Double.POSITIVE_INFINITY;

    // find min/max errors
    for (i = 0; i < data.size(); i++) {
      errd = ((Number) data.get(i)).doubleValue();
      if (errd != null) {
	err = Math.abs(errd.doubleValue());
	if (err < minErr)
	  minErr = err;
	if (err > maxErr)
	  maxErr = err;
      }
    }

    // scale errors
    for (i = 0; i < data.size(); i++) {
      errd = ((Number) data.get(i)).doubleValue();
      if (errd != null) {
	err = Math.abs(errd.doubleValue());
	if (maxErr - minErr > 0) {
	  temp = (((err - minErr) / (maxErr - minErr)) * 20);
	  result.add(new Integer((int) temp));
	}
	else {
	  result.add(new Integer(1));
	}
      }
      else {
	result.add(new Integer(1));
      }
    }

    return result;
  }

  /**
   * Processes the data if necessary.
   */
  protected void process() {
    Capabilities		cap;
    ArrayList<Integer>	scaled;

    if (m_Processed)
	return;

    m_Processed = true;

    createDataset(m_Evaluation);

    try {
	scaled = scale(m_PlotSizes);
	m_PlotSizes = new FastVector();
	m_PlotSizes.addAll(scaled);
    }
    catch (Exception e) {
	e.printStackTrace();
	m_PlotInstances = new Instances(m_PlotInstances, 0);
	m_PlotSizes     = new FastVector();
	m_PlotShapes    = new FastVector();
    }
  }

  /**
   * Returns the underlying Evaluation object.
   *
   * @return		the Evaluation object
   */
  public Evaluation getEvaluation() {
    return m_Evaluation;
  }

  /**
   * Returns the generated dataset that is plotted.
   *
   * @return		the dataset
   */
  public Instances getPlotInstances() {
    process();

    return m_PlotInstances;
  }

  /**
   * Generates a dataset, containing the predicted vs actual values.
   *
   * @param eval	for obtaining the dataset information and predictions
   */
  protected void createDataset(Evaluation eval) {
    ArrayList<Attribute>	atts;
    Attribute			classAtt;
    FastVector		preds;
    int			i;
    double[]			values;
    Instance			inst;
    Prediction		pred;

    m_PlotShapes = new FastVector();
    m_PlotSizes  = new FastVector();
    classAtt     = eval.getHeader().classAttribute();
    preds        = eval.predictions();

    // generate header
    atts     = new ArrayList<Attribute>();
    atts.add(classAtt.copy("predicted" + classAtt.name()));
    atts.add((Attribute) classAtt.copy());
    m_PlotInstances = new Instances(
	eval.getHeader().relationName() + "-classifier_errors", atts, preds.size());
    m_PlotInstances.setClassIndex(m_PlotInstances.numAttributes() - 1);

    // add data
    for (i = 0; i < preds.size(); i++) {
      pred   = (Prediction) preds.elementAt(i);
      values = new double[]{pred.predicted(), pred.actual()};
      inst   = new DenseInstance(pred.weight(), values);
      m_PlotInstances.add(inst);

      if (classAtt.isNominal()) {
        if (weka.core.Utils.isMissingValue(pred.actual()) || weka.core.Utils.isMissingValue(pred.predicted())) {
          m_PlotShapes.addElement(new Integer(Plot2D.MISSING_SHAPE));
        }
        else if (pred.predicted() != pred.actual()) {
          // set to default error point shape
          m_PlotShapes.addElement(new Integer(Plot2D.ERROR_SHAPE));
        }
        else {
          // otherwise set to constant (automatically assigned) point shape
          m_PlotShapes.addElement(new Integer(Plot2D.CONST_AUTOMATIC_SHAPE));
        }
        m_PlotSizes.addElement(new Integer(Plot2D.DEFAULT_SHAPE_SIZE));
      }
      else {
        // store the error (to be converted to a point size later)
        Double errd = null;
        if (!weka.core.Utils.isMissingValue(pred.actual()) && !weka.core.Utils.isMissingValue(pred.predicted())) {
          errd = new Double(pred.predicted() - pred.actual());
          m_PlotShapes.addElement(new Integer(Plot2D.CONST_AUTOMATIC_SHAPE));
        }
        else {
          // missing shape if actual class not present or prediction is missing
          m_PlotShapes.addElement(new Integer(Plot2D.MISSING_SHAPE));
        }
        m_PlotSizes.addElement(errd);
      }
    }
  }

  /**
   * Assembles and returns the plot. The relation name of the dataset gets
   * added automatically.
   *
   * @return			the plot
   * @throws Exception	if plot generation fails
   */
  public PlotData2D getPlotData() throws Exception {
    PlotData2D 	result;

    process();

    result = new PlotData2D(m_PlotInstances);
    result.setShapeSize(m_PlotSizes);
    result.setShapeType(m_PlotShapes);
    result.setPlotName("Classifier Errors" + " (" + m_PlotInstances.relationName() + ")");
    result.addInstanceNumberAttribute();

    return result;
  }
}
