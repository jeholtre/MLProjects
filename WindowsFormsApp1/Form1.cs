using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace WindowsFormsApp1
{
    public partial class Form1 : Form
    {
        public List<Point> points;
        public Form1()
        {
            InitializeComponent();


            
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            
        }

        private void chart1_Click(object sender, EventArgs e)
        {
            string trainFile = "../../../Short.txt";
           // string testFile = "../../../Long.txt";
            string saveFile = "../../../MLMODEL.zip";
            //Model m = new Model(trainFile, testFile, saveFile);
            //m.Train();
            //m.Test();
            Model n = new Model();
            points = new List<Point>();
            for (int i = 1; i < 100; i++)
            {
                n = new Model(trainFile, saveFile, i);
                // Console.Write("The testFraction of {0}", i);
                n.Load();
                points.Add(n.TestGetPoint());

                //Console.WriteLine(points[i - 1].ToString());
            }
            var dataSet = new DataSet();
            foreach (Point p in points)
            {
                chart1.Series["Series1"].Points.AddXY(p.X, p.Y);
            }
            Refresh();
            //   Console.ReadLine();
        }
    }
}
