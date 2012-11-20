/*
 *  (C) 2001 by Argonne National Laboratory
 *      See COPYRIGHT in top-level directory.
 */

/*
 *  @author  Anthony Chan
 */

package viewer.zoomable;

import java.awt.Insets;
import java.awt.Component;
import java.awt.Dialog;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.*;
import javax.swing.border.*;
import java.net.URL;

import base.drawable.TimeBoundingBox;
// import logformat.slog2.LineIDMap;
import viewer.common.Const;
import viewer.common.Parameters;
import viewer.common.SwingWorker;

public class OperationDurationPanel extends JPanel
{
    private static final long                serialVersionUID = 4100;

    private static       Border              Normal_Border = null;

    private              TimeBoundingBox     timebox;
    private              SummarizableView    summarizable;
    private              InitializableDialog summary_dialog;
    private              Dialog              root_dialog;

    public OperationDurationPanel( final TimeBoundingBox   times,
                                   final SummarizableView  summary )
    {
        super();

        timebox         = times;
        summarizable    = summary;
        summary_dialog  = null;
        root_dialog     = null;

        if ( Normal_Border == null ) {
            /*
            Normal_Border = BorderFactory.createCompoundBorder(
                            BorderFactory.createRaisedBevelBorder(),
                            BorderFactory.createLoweredBevelBorder() );
            Normal_Border = BorderFactory.createEtchedBorder();
            */
            Normal_Border = BorderFactory.createEmptyBorder();
        }
        super.setBorder( Normal_Border );

        JButton     stat_btn   = null;
        URL         icon_URL   = getURL( Const.IMG_PATH + "Stat110x40.gif" );
        ImageIcon   icon, icon_shaded;
        Border  raised_border, lowered_border, big_border, huge_border;
        if ( icon_URL != null ) {
            icon        = new ImageIcon( icon_URL );
            icon_shaded = new ImageIcon(
                          GrayFilter.createDisabledImage( icon.getImage() ) );
            stat_btn = new JButton( icon );
            stat_btn.setPressedIcon( icon_shaded );
            raised_border  = BorderFactory.createRaisedBevelBorder();
            lowered_border = BorderFactory.createLoweredBevelBorder();
            big_border = BorderFactory.createCompoundBorder( raised_border,
                                                             lowered_border );
            huge_border = BorderFactory.createCompoundBorder( raised_border,
                                                              big_border );
            stat_btn.setBorder( huge_border );
        }
        else
            stat_btn = new JButton( "Sumary Statistics" );
        stat_btn.setMargin( Const.SQ_BTN2_INSETS );
        stat_btn.setToolTipText(
        "Summary Statistics for the selected duration, timelines & legends" );
        stat_btn.addActionListener( new StatBtnActionListener() );

        // super.add( InfoDialog.BOX_GLUE );
        super.add( stat_btn );
        // super.add( InfoDialog.BOX_GLUE );
    }

    private URL getURL( String filename )
    {
        URL url = null;
        url = getClass().getResource( filename );
        return url;
    }

    private void createSummaryDialog()
    {
        SwingWorker           create_statline_worker;

        root_dialog = (Dialog) SwingUtilities.windowForComponent( this );
        create_statline_worker = new SwingWorker() {
            public Object construct()
            {
                summary_dialog = summarizable.createSummary( root_dialog,
                                                             timebox );
                return null;  // returned value is not needed
            }
            public void finished()
            {
                summary_dialog.pack();
                if ( Parameters.AUTO_WINDOWS_LOCATION ) {
                    Rectangle bounds = root_dialog.getBounds();
                    summary_dialog.setLocation( bounds.x + bounds.width/2,
                                                bounds.y + bounds.height/2 );
                }
                summary_dialog.setVisible( true );
                summary_dialog.init();
            }
        };
        create_statline_worker.start();
    }

    private class StatBtnActionListener implements ActionListener
    {
        public void actionPerformed( ActionEvent evt )
        {
            createSummaryDialog();
        }
    }
}
